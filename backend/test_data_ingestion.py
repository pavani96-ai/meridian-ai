"""Offline ingestion tests: real splitting, mocked PDF loading and vector writes."""
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from exceptions import IngestionError
from rag import data_ingestion


def test_split_long_page_preserves_metadata_and_size():
    page = Document(page_content="payment terms " * 300, metadata={"source": "terms.pdf", "page": 0})
    chunks = data_ingestion.split_into_chunks([page])
    assert len(chunks) > 1
    assert all(0 < len(chunk.page_content) <= data_ingestion.CHUNK_SIZE for chunk in chunks)
    assert all(chunk.metadata == page.metadata for chunk in chunks)


def test_ingest_pdf_sets_sources_and_reuses_stable_ids(monkeypatch):
    pages = [Document(page_content="Payment due in 30 days.", metadata={"page": 0})]
    loader = Mock()
    loader.return_value.load.return_value = pages
    monkeypatch.setattr(data_ingestion, "PyPDFLoader", loader)
    store = Mock()
    monkeypatch.setattr(data_ingestion, "get_vector_store", lambda: store)
    for _ in range(2):
        assert data_ingestion.ingest_pdf("/tmp/input.pdf", "local://terms.pdf") == {"pages": 1, "chunks": 1}
    first, second = store.add_documents.call_args_list
    assert first.kwargs["ids"] == second.kwargs["ids"]
    assert first.args[0][0].metadata == {"page": 0, "source": "local://terms.pdf"}
    assert len(first.kwargs["ids"]) == len(first.args[0])
    loader.assert_called_with("/tmp/input.pdf")


def test_chunk_ids_distinguish_source_page_and_position():
    first = Document(page_content="text", metadata={"source": "a.pdf", "page": 0})
    other_source = Document(page_content="text", metadata={"source": "b.pdf", "page": 0})
    other_page = Document(page_content="text", metadata={"source": "a.pdf", "page": 1})
    assert len({data_ingestion._chunk_id(first, 0), data_ingestion._chunk_id(first, 1),
                data_ingestion._chunk_id(other_source, 0), data_ingestion._chunk_id(other_page, 0)}) == 4


def test_unreadable_pdf_raises_ingestion_error_with_cause(monkeypatch):
    loader = Mock()
    error = ValueError("Invalid PDF")
    loader.return_value.load.side_effect = error
    monkeypatch.setattr(data_ingestion, "PyPDFLoader", loader)
    with pytest.raises(IngestionError, match="broken.pdf") as caught:
        data_ingestion.ingest_pdf("broken.pdf", "local://broken.pdf")
    assert caught.value.__cause__ is error


def test_empty_pdf_skips_vector_store(monkeypatch):
    loader = Mock()
    loader.return_value.load.return_value = []
    monkeypatch.setattr(data_ingestion, "PyPDFLoader", loader)
    store = Mock()
    monkeypatch.setattr(data_ingestion, "get_vector_store", store)
    assert data_ingestion.ingest_pdf("empty.pdf", "local://empty.pdf") == {"pages": 0, "chunks": 0}
    store.assert_not_called()


@pytest.mark.parametrize("files, results, expected", [
    ([], [], {"files": 0, "pages": 0, "chunks": 0}),
    ([("a.pdf", "local://a.pdf"), ("b.pdf", "local://b.pdf")],
     [{"pages": 2, "chunks": 3}, {"pages": 1, "chunks": 2}],
     {"files": 2, "pages": 3, "chunks": 5}),
])
def test_ingest_stored_pdfs_totals(monkeypatch, files, results, expected):
    @contextmanager
    def stored_files():
        yield iter(files)

    monkeypatch.setattr(data_ingestion.document_store, "pdfs", stored_files)
    monkeypatch.setattr(data_ingestion.document_store, "location", lambda: "test-storage")
    ingest = Mock(side_effect=results)
    monkeypatch.setattr(data_ingestion, "ingest_pdf", ingest)
    assert data_ingestion.ingest_stored_pdfs() == expected
    assert [(call.args[0], call.kwargs["source"]) for call in ingest.call_args_list] == files
