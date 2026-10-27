# meridian-ai

### OpenAI chat configuration

Procurement agents, CFO synthesis, and RAG answers use OpenAI chat via `get_llm()`.
Add your own `OPENAI_API_KEY` to the project's `.env` (never commit API keys).
`OPENAI_MODEL_NAME=gpt-4.1-mini` selects the default chat model and
`LLM_REQUEST_TIMEOUT=30` controls request timeout in seconds.
Embeddings now use OpenAI `text-embedding-3-small` at 768 dimensions (`OPENAI_EMBEDDING_MODEL_NAME`). Reindex stored PDFs into the new Chroma collection after migration. Old Google vectors are preserved. In cloud mode, configure a separate 768-dimensional Vertex index and reindex before querying; never mix old and new embedding vectors.
Restart the backend after changing environment variables.
