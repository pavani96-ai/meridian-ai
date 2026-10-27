"""Offline tests of OpenAI embedding configuration."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from exceptions import ConfigurationError


@pytest.mark.parametrize('cloud', [False, True])
def test_openai_embeddings_config_and_missing_key(cloud):
    config = ModuleType('config.settings')
    config.settings = SimpleNamespace(is_cloud=cloud, OPENAI_API_KEY='test-not-a-real-key', embedding_model_name='text-embedding-3-small', embedding_dimensions=768, llm_request_timeout=30)
    spec = importlib.util.spec_from_file_location('isolated_embeddings', Path(__file__).parent / 'rag/embeddings.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'config.settings': config}):
        spec.loader.exec_module(module)
    with patch.object(module, 'OpenAIEmbeddings') as factory:
        assert module.get_embeddings() is factory.return_value
        assert module.get_embeddings() is factory.return_value
        factory.assert_called_once_with(model='text-embedding-3-small', api_key='test-not-a-real-key', dimensions=768, request_timeout=30, max_retries=0)
        module.get_embeddings.cache_clear()
        config.settings.OPENAI_API_KEY = ''
        with pytest.raises(ConfigurationError, match='OPENAI_API_KEY'):
            module.get_embeddings()
        assert factory.call_count == 1
