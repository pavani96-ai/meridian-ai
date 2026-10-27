"""
FastAPI backend — Procurement Audit & RAG Document Intelligence
================================================================
Run locally (from the project root):
    uvicorn api.main:app --app-dir backend --reload --port 8080
Interactive API docs: http://localhost:8080/docs

Exposes REST endpoints for:
  1. Running multi-agent procurement audits       (agent/agents.py)
  2. Uploading documents into the vector store       (rag/data_ingestion.py)
     APP_MODE=local -> Chroma on disk, APP_MODE=cloud -> Vertex AI Vector Search
  3. Querying the RAG pipeline                    (rag/retrieval.py)
In Docker / Cloud Run the same server also serves the built React app.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.endpoints import agent_router, health_router, rag_router, status_router
from config.settings import settings
from exceptions import ConfigurationError, IngestionError
from logger import _LOGGER_INSTANCE
from logger import GLOBAL_LOGGER as log


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log startup and flush buffered logs when the server shuts down."""
    log.info("Meridian AI starting", mode=settings.app_mode, llm=settings.llm_model_name)
    try:
        yield  # The application handles requests here.
    finally:
        _LOGGER_INSTANCE.flush_to_gcs()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-agent procurement audit & RAG document intelligence backend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ConfigurationError)
async def configuration_error(_: Request, exc: ConfigurationError):
    """Return HTTP 503 when required application configuration is missing."""
    log.error("Configuration error", error=str(exc))
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(IngestionError)
async def ingestion_error(_: Request, exc: IngestionError):
    """Return HTTP 422 when a document cannot be ingested."""
    log.error("Ingestion error", error=str(exc))
    return JSONResponse(status_code=422, content={"detail": str(exc)})


app.include_router(health_router)
app.include_router(status_router)
app.include_router(agent_router)
app.include_router(rag_router)


# ============================================================
# Serve the React frontend (only exists after `npm run build`, e.g. inside Docker)
# ============================================================

frontend_dist = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../frontend/dist"))

if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    # Catch-all: serve a real file if it exists, otherwise index.html (React handles the route)
    @app.get("/{catchall:path}", include_in_schema=False)
    def serve_frontend(catchall: str):
        if catchall.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = os.path.realpath(os.path.join(frontend_dist, catchall))
        if file_path.startswith(frontend_dist + os.sep) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    log.warning("Frontend dist not found — frontend will not be served.", path=frontend_dist)
