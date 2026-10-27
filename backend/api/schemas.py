"""Request and response models for procurement audits and RAG queries."""
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# --- Agent schemas ---


class AuditRequest(BaseModel):
    request_text: NonEmptyText = Field(description="Procurement request to audit.")


class AuditResponse(BaseModel):
    risk_result: str
    tax_result: str
    control_result: str
    cfo_memo: str


# --- RAG schemas ---


class QueryRequest(BaseModel):
    query: NonEmptyText
    retriever_type: Literal["similarity", "multiquery", "contextual"] = "similarity"


class QueryResponse(BaseModel):
    answer: str
