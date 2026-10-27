"""Offline supervisor tests: no .env, API keys, or external calls required.

Run: .venv/bin/python -m pytest backend/test_agents.py -v
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch, sentinel

import pytest
from pydantic import BaseModel
from langchain.agents.structured_output import ProviderStrategy


@pytest.fixture
def agents():
    folder = Path(__file__).parent / 'agent'
    package = ModuleType('_offline_agents')
    package.__path__ = [str(folder)]
    tools = ModuleType('agent.tools')
    for name in ('web_search', 'check_sanctions_list', 'get_vendor_credit_score',
                 'calculate_cross_border_tax', 'validate_fx_hedge', 'categorize_expense'):
        setattr(tools, name, Mock(name=name))
    llm = ModuleType('rag.llm')
    llm.get_llm = Mock(side_effect=AssertionError('Unexpected live model initialization'))
    llm.extract_text = Mock(side_effect=lambda content: content)
    logger = ModuleType('logger')
    logger.GLOBAL_LOGGER = Mock()
    middleware = ModuleType('_offline_agents.middleware')
    middleware.build_agent_middleware = Mock(return_value=[sentinel.middleware])
    replacements = {'_offline_agents': package, 'agent.tools': tools, 'rag.llm': llm,
                    'logger': logger, '_offline_agents.middleware': middleware}
    with patch.dict(sys.modules, replacements):
        spec = importlib.util.spec_from_file_location('_offline_agents.agents', folder / 'agents.py')
        module = importlib.util.module_from_spec(spec)
        # Import must not instantiate agents or run the sample audit.
        spec.loader.exec_module(module)
        module.get_llm.assert_not_called()
        with patch.object(module, 'create_agent', side_effect=AssertionError('Unmocked agent creation')):
            yield module


@pytest.mark.parametrize('provided_model', [True, False])
def test_factory_configures_structured_output_and_middleware(agents, provided_model):
    agents.get_llm.side_effect = None
    agents.get_llm.return_value = sentinel.model
    agents.create_agent.side_effect = None
    agents.create_agent.return_value = sentinel.agent
    kwargs = {'model': sentinel.model} if provided_model else {}
    result = agents.create_specialized_agent([sentinel.tool], 'Test prompt', agents.RiskAssessment, **kwargs)
    assert result is sentinel.agent
    config = agents.create_agent.call_args.kwargs
    assert config['model'] is sentinel.model
    assert config['tools'] == [sentinel.tool]
    assert config['system_prompt'] == 'Test prompt'
    assert config['middleware'] == [sentinel.middleware]
    assert isinstance(config['response_format'], ProviderStrategy)
    assert config['response_format'].schema is agents.RiskAssessment
    assert agents.get_llm.call_count == (0 if provided_model else 1)


def test_supervisor_shares_model_and_assigns_specialists(agents):
    agents.get_llm.side_effect = None
    agents.get_llm.return_value = sentinel.model
    with patch.object(agents, 'create_specialized_agent', side_effect=[sentinel.risk, sentinel.tax, sentinel.control]) as factory:
        supervisor = agents.ProcurementSupervisor()
    agents.get_llm.assert_called_once_with()
    assert supervisor.llm is sentinel.model
    assert (supervisor.risk_agent, supervisor.tax_agent, supervisor.control_agent) == (sentinel.risk, sentinel.tax, sentinel.control)
    assert factory.call_args_list == [
        call(tools=[agents.check_sanctions_list, agents.get_vendor_credit_score], system_prompt=agents.RISK_AGENT_PROMPT, response_schema=agents.RiskAssessment, model=sentinel.model),
        call(tools=[agents.calculate_cross_border_tax, agents.validate_fx_hedge], system_prompt=agents.TAX_AGENT_PROMPT, response_schema=agents.TaxAssessment, model=sentinel.model),
        call(tools=[agents.categorize_expense], system_prompt=agents.CONTROL_AGENT_PROMPT, response_schema=agents.ControlAssessment, model=sentinel.model),
    ]


class Report(BaseModel):
    verdict: str


def test_invoke_serializes_validated_response(agents):
    agent = Mock()
    agent.invoke.return_value = {'structured_response': Report(verdict='APPROVED')}
    result = agents.ProcurementSupervisor._invoke_agent(agent, 'Review invoice')
    assert json.loads(result) == {'verdict': 'APPROVED'}
    agent.invoke.assert_called_once_with({'messages': [{'role': 'user', 'content': 'Review invoice'}]})


@pytest.mark.parametrize('request_text', ['', '   '])
def test_invoke_rejects_empty_request(agents, request_text):
    agent = Mock()
    with pytest.raises(ValueError, match='procurement request'):
        agents.ProcurementSupervisor._invoke_agent(agent, request_text)
    agent.invoke.assert_not_called()


@pytest.mark.parametrize('result', [{}, {'structured_response': None}, {'structured_response': {'verdict': 'APPROVED'}}, {'messages': []}])
def test_invoke_rejects_missing_validated_output(agents, result):
    agent = Mock()
    agent.invoke.return_value = result
    with pytest.raises(ValueError, match='validated structured response'):
        agents.ProcurementSupervisor._invoke_agent(agent, 'Review invoice')


@pytest.fixture
def supervisor(agents):
    instance = agents.ProcurementSupervisor.__new__(agents.ProcurementSupervisor)
    for field, verdict in [('risk_agent', 'risk report'), ('tax_agent', 'tax report'), ('control_agent', 'control report')]:
        agent = Mock()
        agent.invoke.return_value = {'structured_response': Report(verdict=verdict)}
        setattr(instance, field, agent)
    instance.llm = Mock()
    instance.llm.invoke.return_value = SimpleNamespace(content='CFO memorandum')
    return instance


def test_audit_sequence_synthesis_outputs_and_phase_logs(agents, supervisor):
    timeline = Mock()
    for name in ('risk_agent', 'tax_agent', 'control_agent', 'llm'):
        timeline.attach_mock(getattr(supervisor, name).invoke, name)
    timeline.attach_mock(agents.log.info, 'log')
    result = supervisor.run_audit('Review invoice')
    assert [c[0] for c in timeline.mock_calls] == [
        'log', 'risk_agent', 'log', 'log', 'tax_agent', 'log',
        'log', 'control_agent', 'log', 'log', 'llm', 'log',
    ]
    expected = {key: Report(verdict=value).model_dump_json() for key, value in [
        ('risk_result', 'risk report'), ('tax_result', 'tax report'), ('control_result', 'control report')]}
    assert result == {**expected, 'cfo_memo': 'CFO memorandum'}
    supervisor.llm.invoke.assert_called_once_with(agents.SYNTHESIS_PROMPT_TEMPLATE.format(**expected))
    agents.extract_text.assert_called_once_with('CFO memorandum')
    for name in ('risk_agent', 'tax_agent', 'control_agent'):
        getattr(supervisor, name).invoke.assert_called_once_with({'messages': [{'role': 'user', 'content': 'Review invoice'}]})
    assert agents.log.info.call_args_list[-1] == call('Audit phase complete', phase='4 - CFO Synthesis')


@pytest.mark.parametrize('phase', ['risk_agent', 'tax_agent', 'control_agent'])
def test_failed_specialist_stops_later_phases(agents, supervisor, phase):
    failure = RuntimeError('Mock provider unavailable')
    getattr(supervisor, phase).invoke.side_effect = failure
    with pytest.raises(RuntimeError) as exc:
        supervisor.run_audit('Review invoice')
    assert exc.value is failure
    order = ['risk_agent', 'tax_agent', 'control_agent', 'llm']
    for name in order[order.index(phase) + 1:]:
        getattr(supervisor, name).invoke.assert_not_called()
    assert call('Audit phase complete', phase='4 - CFO Synthesis') not in agents.log.info.call_args_list


@pytest.mark.parametrize('content', ['', '  ', None])
def test_empty_cfo_memo_is_rejected(agents, supervisor, content):
    supervisor.llm.invoke.return_value = SimpleNamespace(content=content)
    with pytest.raises(ValueError, match='empty memorandum'):
        supervisor.run_audit('Review invoice')
    assert call('Audit phase complete', phase='4 - CFO Synthesis') not in agents.log.info.call_args_list


def test_cfo_provider_exception_propagates(agents, supervisor):
    supervisor.llm.invoke.side_effect = RuntimeError('Mock synthesis failure')
    with pytest.raises(RuntimeError, match='Mock synthesis failure'):
        supervisor.run_audit('Review invoice')
