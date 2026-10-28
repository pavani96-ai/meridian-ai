"""REST endpoints. Each route is thin: validate input, call the real logic, return JSON.

    /api/health        - is the server up?
    /api/status        - configuration summary shown on the System Status tab
    /api/agent/audit   - run the multi-agent procurement audit        (agent/agents.py)
    /api/agent/audit/stream - stream audit progress and CFO memo as NDJSON
    /api/rag/ask       - ask a question about the ingested documents  (rag/retrieval.py)
    /api/rag/upload    - upload PDFs and index them                   (rag/data_ingestion.py)
    /api/rag/ingest    - index every PDF already in the document store
                         (local folder or GCS bucket, depending on APP_MODE)

Errors: our own exceptions (exceptions/) are turned into clear HTTP answers in api/main.py,
so the routes only catch what they can explain better themselves.
"""

import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import AuditRequest, AuditResponse, QueryRequest, QueryResponse
from agent.agents import ProcurementSupervisor
from config.settings import settings
from exceptions import MeridianError, PromptInjectionError
from prompt_security import PROMPT_INJECTION_REFUSAL
from logger import GLOBAL_LOGGER as log
from rag import document_store
from rag.data_ingestion import ingest_pdf, ingest_stored_pdfs
from rag.retrieval import ask_llm as ask_question


# In-memory upload tracker; resets on restart.
_upload_history: list[dict] = []
_server_start_time = datetime.now(timezone.utc)


health_router = APIRouter(prefix="/api", tags=["health"])
status_router = APIRouter(prefix="/api", tags=["status"])
agent_router = APIRouter(prefix="/api/agent", tags=["agent"])
rag_router = APIRouter(prefix="/api/rag", tags=["rag"])


@lru_cache(maxsize=1)
def get_supervisor() -> ProcurementSupervisor:
    """Create the procurement supervisor on first use and reuse it per process."""
    return ProcurementSupervisor()


@health_router.get("/health")
def health():
    """Return server health and the configured application mode."""
    return {"status": "ok", "mode": settings.app_mode}


@status_router.get("/status")
def system_status():
    """Return system configuration and health information (makes no calls to Google)."""
    uptime_seconds = int((datetime.now(timezone.utc) - _server_start_time).total_seconds())
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    cloud = settings.is_cloud

    return {
        "mode": settings.app_mode,
        "backend": {
            "status": "healthy",
            "uptime": f"{hours:02d}h {minutes:02d}m {seconds:02d}s",
            "python_version": platform.python_version(),
            "platform": platform.system(),
        },
        "gcp": {
            "project_id": settings.GCP_PROJECT if cloud else "not used (local mode)",
            "region": settings.GCP_REGION if cloud else "-",
        },
        "storage": {
            "location": document_store.location(),
            "gcs_bucket": settings.GCS_BUCKET_NAME if cloud else "-",
            "gcs_prefix": settings.GCS_PREFIX if cloud else "-",
        },
        "vector_search": {
            "backend": "Vertex AI Vector Search" if cloud else "Chroma (local disk)",
            "index_id": (settings.vector_search_index_id or "NA") if cloud else settings.chroma_dir,
            "endpoint_id": (settings.vector_search_index_endpoint_id or "NA") if cloud else "-",
            "stream_update": cloud,
        },
        "models": {
            "embedding": settings.embedding_model_name,
            "llm": settings.llm_model_name,
            "llm_framework": (
                "OpenAI (chat + embeddings) via LangChain v1"
            ),
        },
        "ingestion": {
            "uploads_this_session": len(_upload_history),
        },
    }



# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@agent_router.post("/audit", response_model=AuditResponse)
def run_audit(payload: AuditRequest):
    """Run the multi-agent procurement audit and return all phase results."""
    try:
        return AuditResponse(**get_supervisor().run_audit(payload.request_text))
    except MeridianError:
        raise  # Handled centrally in api/main.py.
    except Exception as e:
        log.exception("Audit failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@agent_router.post("/audit/stream")
def stream_audit(payload: AuditRequest):
    """Stream newline-delimited JSON; errors after headers are terminal events."""
    supervisor = get_supervisor()

    def events():
        try:
            for event in supervisor.stream_audit(payload.request_text):
                yield json.dumps(event) + "\n"
        except PromptInjectionError:
            yield json.dumps({"type": "error", "code": "prompt_injection_detected",
                              "message": PROMPT_INJECTION_REFUSAL}) + "\n"
        except Exception:
            log.exception("Streaming audit failed")
            yield json.dumps({"type": "error", "code": "audit_failed",
                              "message": "The audit could not complete. Please try again."}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


@rag_router.post("/ask", response_model=QueryResponse)
def rag_query(payload: QueryRequest):
    """Answer a question using the indexed documents."""
    try:
        return QueryResponse(answer=ask_question(payload.query, payload.retriever_type))
    except MeridianError:
        raise  # Handled centrally in api/main.py.
    except Exception as e:
        log.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@rag_router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(default=[])):
    """Store uploaded PDFs, then chunk, embed, and index them."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    tmp_dir = tempfile.mkdtemp()
    try:
        for upload in files:
            filename = os.path.basename((upload.filename or "").replace("\\", "/"))
            if not filename.lower().endswith(".pdf"):
                results.append({"filename": filename, "status": "skipped",
                                "reason": "Only PDF files are supported."})
                continue

            # Copy the spooled upload to disk without reading the entire PDF into memory.
            local_path = os.path.join(tmp_dir, filename)
            with open(local_path, "wb") as target:
                await run_in_threadpool(shutil.copyfileobj, upload.file, target)

            # Keep the original PDF (local folder or GCS). Indexing still works if this fails.
            try:
                await run_in_threadpool(document_store.save, local_path, filename)
            except Exception as store_err:
                log.warning("Could not store original file", filename=filename, error=str(store_err))

            # ingest_pdf is slow and blocking, so run it in a worker thread to keep the server responsive.
            summary = await run_in_threadpool(ingest_pdf, local_path, f"upload://{filename}")
            if summary["chunks"] == 0:
                results.append({"filename": filename, "status": "skipped",
                                "reason": "No extractable text found in PDF (may be scanned)."})
                continue
            log.info("Successfully ingested chunks", filename=filename, **summary)
            results.append({"filename": filename, "status": "ingested", **summary})
    except MeridianError:
        raise  # Handled centrally in api/main.py.
    except Exception as e:
        log.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _upload_history.extend(results)
        for upload in files:
            await upload.close()

    return {"results": results}


@rag_router.get("/uploads")
def list_uploads():
    """Return the in-memory upload history; resets on restart."""
    return {"uploads": _upload_history}


@rag_router.post("/ingest")
@rag_router.post("/ingest_gcs", include_in_schema=False)
def trigger_store_ingestion():
    """Index every PDF already in the document store (local folder or GCS bucket)."""
    try:
        totals = ingest_stored_pdfs()
        return {"status": f"Ingestion complete ({settings.app_mode} mode)", **totals}
    except MeridianError:
        raise  # Handled centrally in api/main.py.
    except Exception as e:
        log.exception("Stored-document ingestion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
