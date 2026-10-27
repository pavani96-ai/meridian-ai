"""Offline middleware checks without loading credentials."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import ToolMessage


@pytest.fixture
def middleware():
    config = ModuleType('config.settings')
    config.settings = SimpleNamespace(agent_max_model_calls=8, agent_max_tool_calls=6, agent_max_retries=2)
    logger = ModuleType('logger')
    logger.GLOBAL_LOGGER = Mock()
    spec = importlib.util.spec_from_file_location('isolated_middleware', Path(__file__).parent / 'agent/middleware.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'config.settings': config, 'logger': logger}):
        spec.loader.exec_module(module)
    return module


def test_stack_builds_fresh_instances(middleware):
    first, second = middleware.build_agent_middleware(), middleware.build_agent_middleware()
    assert len(first) == 7
    assert first[3] is second[3] is middleware.log_tool_calls
    assert all(a is not b for i, (a, b) in enumerate(zip(first, second)) if i != 3)
    assert [type(m).__name__ for m in first] == ['PIIMiddleware', 'PIIMiddleware', 'ModelCallLimitMiddleware', 'ToolCallLoggingMiddleware', 'ToolCallLimitMiddleware', 'ModelRetryMiddleware', 'ToolRetryMiddleware']


def test_logs_success_without_argument_values(middleware):
    request = SimpleNamespace(tool_call={'name': 'test', 'args': {'email': 'private@example.com'}})
    result = ToolMessage(content='done', tool_call_id='1')
    assert middleware.log_tool_calls.wrap_tool_call(request, lambda _: result) is result
    kwargs = middleware.log.info.call_args.kwargs
    assert kwargs['outcome'] == 'success'
    assert kwargs['duration_ms'] >= 0
    assert kwargs['args'] == {'email': '[REDACTED]'}
    assert 'private@example.com' not in str(middleware.log.mock_calls)


def test_logs_and_reraises_exception(middleware):
    def fail(_):
        raise RuntimeError('private error')
    with pytest.raises(RuntimeError):
        middleware.log_tool_calls.wrap_tool_call(SimpleNamespace(tool_call={'name': 'test'}), fail)
    assert middleware.log.info.call_args.kwargs['outcome'] == 'exception'
    assert 'private error' not in str(middleware.log.mock_calls)


def test_async_logging(middleware):
    result = ToolMessage(content='failed', tool_call_id='1', status='error')
    async def handler(_):
        return result
    request = SimpleNamespace(tool_call={'name': 'test'})
    assert asyncio.run(middleware.log_tool_calls.awrap_tool_call(request, handler)) is result
    assert middleware.log.info.call_args.kwargs['outcome'] == 'error'
