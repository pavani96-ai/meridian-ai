"""Offline checks for the shared OpenAI chat factory."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from exceptions import ConfigurationError


@pytest.fixture
def llm_module():
    config = ModuleType('config.settings')
    config.settings = SimpleNamespace(OPENAI_API_KEY='test-not-a-real-key', llm_model_name='gpt-4.1-mini', llm_temperature=0, llm_request_timeout=30)
    spec = importlib.util.spec_from_file_location('isolated_llm', Path(__file__).parent / 'rag/llm.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'config.settings': config}):
        spec.loader.exec_module(module)
    with patch.object(module, 'ChatOpenAI') as factory:
        yield module, factory
    module.get_llm.cache_clear()


def test_openai_configuration_and_cache(llm_module):
    module, factory = llm_module
    assert module.get_llm() is factory.return_value
    assert module.get_llm() is factory.return_value
    factory.assert_called_once_with(model='gpt-4.1-mini', api_key='test-not-a-real-key', temperature=0, timeout=30, max_retries=0)


def test_missing_key_fails_clearly(llm_module):
    module, factory = llm_module
    module.settings.OPENAI_API_KEY = ''
    with pytest.raises(ConfigurationError, match='OPENAI_API_KEY is empty'):
        module.get_llm()
    factory.assert_not_called()


def test_extracts_text_blocks(llm_module):
    module, _ = llm_module
    assert module.extract_text('OK') == 'OK'
    assert module.extract_text([{'type': 'text', 'text': 'Hello'}, {'type': 'image'}, {'type': 'text', 'text': 'world'}]) == 'Hello\nworld'
