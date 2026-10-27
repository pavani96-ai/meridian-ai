"""OpenAI embeddings shared by document indexing and retrieval in both app modes.

Use a fresh Chroma collection / Vertex index when changing embedding models;
matching dimensions alone do not make vectors from different models compatible.
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from exceptions import ConfigurationError


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Create the embedding client lazily, using the same model for reads and writes."""
    if not settings.OPENAI_API_KEY:
        raise ConfigurationError("OPENAI_API_KEY is empty. Add it to .env for embeddings.")
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=settings.OPENAI_API_KEY,
        dimensions=settings.embedding_dimensions,
        request_timeout=settings.llm_request_timeout,
        max_retries=0,
    )
