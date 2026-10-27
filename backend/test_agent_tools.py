"""Offline tool tests. Run from meridian-ai: .venv/bin/python -m pytest backend/test_agent_tools.py -v"""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def tools():
    # Isolate settings and LLM initialization: never read .env or contact Gemini.
    settings_module = ModuleType('config.settings')
    settings_module.settings = SimpleNamespace(TAVILY_API_KEY='test-key')
    llm_module = ModuleType('rag.llm')
    llm_module.get_llm = Mock(side_effect=AssertionError('Unexpected live LLM call'))
    llm_module.extract_text = lambda content: content
    logger_module = ModuleType('logger')
    logger_module.GLOBAL_LOGGER = Mock()
    spec = importlib.util.spec_from_file_location('isolated_agent_tools', Path(__file__).parent / 'agent' / 'tools.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'config.settings': settings_module, 'rag.llm': llm_module, 'logger': logger_module}):
        spec.loader.exec_module(module)
    with patch.object(module, 'TavilySearch') as search, patch.object(module.requests, 'get') as get, patch.object(module, '_ask_llm') as llm:
        search.side_effect = AssertionError('Unexpected search call')
        get.side_effect = AssertionError('Unexpected HTTP call')
        llm.side_effect = AssertionError('Unexpected LLM call')
        module.search_mock, module.http_mock, module.llm_mock = search, get, llm
        yield module


def evidence(tools, records=None):
    tools.web_search = SimpleNamespace(invoke=Mock(return_value={
        'results': records if records is not None else [{'url': 'https://gov.uk/example', 'content': 'Test evidence'}],
        'checked_at': '2026-10-27T00:00:00+00:00',
    }))


def answer(tools, value):
    tools.llm_mock.side_effect = None
    tools.llm_mock.return_value = value


def test_web_search_filters_untrusted_domains(tools):
    tools.search_mock.side_effect = None
    tools.search_mock.return_value.invoke.return_value = {'results': [
        {'url': 'https://gov.uk/list', 'content': 'official'},
        {'url': 'https://gov.uk.evil.example/list', 'content': 'untrusted'},
    ]}
    result = tools.web_search.invoke({'query': 'vendor', 'include_domains': ['gov.uk']})
    assert [r['url'] for r in result['results']] == ['https://gov.uk/list']


def test_web_search_missing_key(tools):
    tools.settings.TAVILY_API_KEY = ''
    assert 'error' in tools.web_search.invoke({'query': 'vendor'})
    tools.search_mock.assert_not_called()


def test_sanctions_without_evidence_is_inconclusive(tools):
    evidence(tools, [])
    assert tools.check_sanctions_list.invoke({'vendor_name': 'Example'}).startswith('INCONCLUSIVE:')
    tools.llm_mock.assert_not_called()


def test_sanctions_includes_sources(tools):
    evidence(tools)
    answer(tools, 'POTENTIAL MATCH: Identity confirmation required.')
    result = tools.check_sanctions_list.invoke({'vendor_name': 'Example'})
    assert 'POTENTIAL MATCH' in result and 'https://gov.uk/example' in result


@pytest.mark.parametrize('score,risk', [(80, 'LOW'), (60, 'MODERATE'), (45, 'HIGH')])
def test_credit_score(tools, score, risk):
    evidence(tools)
    answer(tools, json.dumps({'score': score, 'risk_level': risk, 'reasoning': 'Mock evidence assessment'}))
    result = tools.get_vendor_credit_score.invoke({'vendor_name': 'Example'})
    assert f'{score}/100 | RISK: {risk}' in result


def test_credit_rejects_inconsistent_risk(tools):
    evidence(tools)
    answer(tools, json.dumps({'score': 80, 'risk_level': 'HIGH', 'reasoning': 'Invalid'}))
    assert 'UNAVAILABLE' in tools.get_vendor_credit_score.invoke({'vendor_name': 'Example'})


@pytest.mark.parametrize('includes_duty,vat,total', [(True, 220, 1320), (False, 200, 1300)])
def test_tax_calculated_locally(tools, includes_duty, vat, total):
    evidence(tools)
    answer(tools, json.dumps({'vat_gst_rate': .2, 'import_duty_rate': .1,
        'vat_includes_duty': includes_duty, 'trade_agreement_notes': 'Mock assumptions and evidence',
        'tax_amount': 999999}))
    result = tools.calculate_cross_border_tax.invoke({'amount': 1000, 'origin': 'A', 'destination': 'B'})
    assert result['import_duty_amount'] == 100
    assert result['vat_gst_amount'] == vat
    assert result['total_cost'] == total


def test_tax_invalid_llm_response(tools):
    evidence(tools)
    answer(tools, 'not JSON')
    assert 'error' in tools.calculate_cross_border_tax.invoke({'amount': 1000, 'origin': 'A', 'destination': 'B'})


@pytest.mark.parametrize('rate,status', [(100, 'FX SUCCESS'), (105, 'FX SUCCESS'), (106, 'FX ALERT')])
def test_fx_variance_boundary(tools, rate, status):
    tools.http_mock.side_effect = None
    tools.http_mock.return_value.json.return_value = {'rate': 100, 'date': '2026-10-27'}
    assert tools.validate_fx_hedge.invoke({'currency_pair': 'USD/INR', 'rate_used': rate}).startswith(status)


def test_fx_llm_fallback(tools):
    import requests
    tools.http_mock.side_effect = requests.RequestException('Offline')
    answer(tools, '100')
    result = tools.validate_fx_hedge.invoke({'currency_pair': 'USD_INR', 'rate_used': 100})
    assert 'LLM-estimated' in result and 'Manual Treasury review' in result


def test_expense_success(tools):
    expected = 'CLASSIFICATION: CapEx\nREASONING: Server\nDEPRECIATION: 3 years\nFLAGS: None'
    answer(tools, expected)
    assert tools.categorize_expense.invoke({'amount': 1000, 'item_description': 'server'}) == expected


@pytest.mark.parametrize('amount,description', [(50001, 'server'), (-1, 'server'), (float('nan'), 'server'), (100, '')])
def test_expense_rejected_before_llm(tools, amount, description):
    assert tools.categorize_expense.invoke({'amount': amount, 'item_description': description}).startswith('ERROR:')
    tools.llm_mock.assert_not_called()


@pytest.mark.parametrize('amount,description,classification', [(100, 'server', 'CapEx'), (100, 'machine', 'CapEx'), (100, 'supplies', 'OpEx'), (50000, 'purchase', 'CapEx')])
def test_expense_fallback(tools, amount, description, classification):
    answer(tools, 'LLM_ERROR: Gemini unavailable')
    result = tools.categorize_expense.invoke({'amount': amount, 'item_description': description})
    assert result.startswith(f'CLASSIFICATION: {classification}')
    assert 'Fallback:' in result and 'Manual financial review' in result
