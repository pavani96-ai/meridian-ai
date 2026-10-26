from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings
from exceptions import ConfigurationError

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """ Create the chat model on first use (not at import time) and reuse it afterwards"""
    if not settings.GOOGLE_API_KEY:
        raise ConfigurationError("Google_API_KEY is empty. Add it to .env (https://aistudio.google.com/apikey).")
    return ChatGoogleGenerativeAI(
        model = settings.llm_model_name,
        google_api_key = settings.GOOGLE_API_KEY,
        temperature = settings.llm_temperature,
    )

def extract_text(content) -> str:
    """ gemini may return a list of content blocks , e.g, [{'type': text' , 'text': '...'}]; join the text.""" 
    if isinstance(content, list):
        return "\n".join(block.get("text","") for block in content if isinstance(block,dict) and block.get("type")=='text') 
    return content
