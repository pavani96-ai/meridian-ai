"""Exercise the real stream route with offline dependencies, without app credentials."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
import pytest

from api.schemas import AuditRequest
from exceptions import PromptInjectionError
from prompt_security import PROMPT_INJECTION_REFUSAL


@pytest.fixture
def client():
    # Compile only the route so imports cannot initialize provider/storage clients.
    source = ast.parse((Path(__file__).parent / 'api/endpoints.py').read_text())
    route = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'stream_audit')
    supervisor = Mock()
    namespace = dict(agent_router=APIRouter(), AuditRequest=AuditRequest,
                     StreamingResponse=StreamingResponse, get_supervisor=lambda: supervisor,
                     json=json, PromptInjectionError=PromptInjectionError,
                     PROMPT_INJECTION_REFUSAL=PROMPT_INJECTION_REFUSAL, log=Mock())
    exec(compile(ast.Module(body=[route], type_ignores=[]), '<stream-route>', 'exec'), namespace)
    app = FastAPI()
    app.include_router(namespace['agent_router'])
    with TestClient(app) as session:
        yield session, supervisor


def test_stream_route_returns_delimited_events(client):
    session, supervisor = client
    events = [{'type': 'progress', 'message': 'Reviewing…'}, {'type': 'memo_delta', 'text': 'Line 1\nLine 2'},
              {'type': 'complete', 'audit': {'cfo_memo': 'Line 1\nLine 2'}}]
    supervisor.stream_audit.return_value = iter(events)
    with session.stream('POST', '/audit/stream', json={'request_text': 'Review invoice'}) as response:
        assert response.headers['content-type'].startswith('application/x-ndjson')
        assert response.headers['x-accel-buffering'] == 'no'
        assert [json.loads(line) for line in response.iter_lines()] == events


@pytest.mark.parametrize('error,code', [(PromptInjectionError('secret'), 'prompt_injection_detected'),
                                      (RuntimeError('secret'), 'audit_failed')])
def test_stream_route_emits_safe_terminal_error_after_progress(client, error, code):
    session, supervisor = client
    def fail(_):
        yield {'type': 'progress', 'message': 'Starting'}
        raise error
    supervisor.stream_audit.side_effect = fail
    response = session.post('/audit/stream', json={'request_text': 'Review invoice'})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['type'] == 'error'
    assert events[-1]['code'] == code
    assert 'secret' not in response.text
    if code == 'prompt_injection_detected':
        assert events[-1]['message'] == PROMPT_INJECTION_REFUSAL


def test_empty_input_is_rejected_before_streaming(client):
    session, supervisor = client
    assert session.post('/audit/stream', json={'request_text': ' '}).status_code == 422
    supervisor.stream_audit.assert_not_called()
