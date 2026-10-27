from functools import lru_cache

from langchain_openai import ChatOpenAI

from config.settings import settings
from exceptions import ConfigurationError

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """ Create the chat model on first use (not at import time) and reuse it afterwards"""
    if not settings.OPENAI_API_KEY:
        raise ConfigurationError("OPENAI_API_KEY is empty. Add it to .env.")
    return ChatOpenAI(
        model = settings.llm_model_name,
        api_key = settings.OPENAI_API_KEY,
        temperature = settings.llm_temperature,
        timeout=settings.llm_request_timeout,
        # Agent middleware owns retry/backoff instead of nesting retry loops.
        max_retries=0,
    )

def extract_text(content) -> str:
    """Extract text from a string or a provider's list of text content blocks."""
    if isinstance(content, list):
        return "\n".join(block.get("text","") for block in content if isinstance(block,dict) and block.get("type")=='text') 
    return content
