"""Offline tests for retrieval strategies and the complete LCEL answer pipeline."""
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag import retrieval


def test_similarity_passes_query_and_limit(monkeypatch):
    store = Mock()
    store.similarity_search.return_value = [Document(page_content="Payment terms")]
    monkeypatch.setattr(retrieval, "get_vector_store", lambda: store)
    assert retrieval._similarity("terms?", k=5) == store.similarity_search.return_value
    store.similarity_search.assert_called_once_with("terms?", k=5)


def test_unknown_strategy_falls_back_to_similarity(monkeypatch):
    search = Mock(return_value=[])
    monkeypatch.setitem(retrieval.STRATEGIES, "similarity", search)
    assert retrieval.build_retriever("unknown").invoke("question") == []
    search.assert_called_once_with("question")


def test_multiquery_merges_duplicate_documents_and_queries(monkeypatch):
    doc = Document(page_content="Terms", metadata={"source": "terms.pdf"})
    rewriter = Mock()
    rewriter.invoke.return_value = retrieval.AlternativeQuestions(
        questions=["original", "rewrite", "rewrite"]
    )
    llm = Mock()
    llm.with_structured_output.return_value = rewriter
    monkeypatch.setattr(retrieval, "get_llm", lambda: llm)
    search = Mock(return_value=[doc, doc])
    monkeypatch.setattr(retrieval, "_similarity", search)
    assert retrieval._multiquery("original") == [doc]
    assert [call.args[0] for call in search.call_args_list] == ["original", "rewrite"]


def test_contextual_filters_irrelevant_chunks_and_preserves_sources(monkeypatch):
    docs = [Document(page_content=str(i), metadata={"source": f"{i}.pdf"}) for i in range(3)]
    search = Mock(return_value=docs)
    monkeypatch.setattr(retrieval, "_similarity", search)

    def extract(prompt):
        content = prompt.to_messages()[-1].content
        return AIMessage(content={"0": "  Pay in 30 days.  ", "1": "NO_OUTPUT", "2": " "}[content[-1]])

    monkeypatch.setattr(retrieval, "get_llm", lambda: RunnableLambda(extract))
    assert retrieval._contextual("terms?") == [
        Document(page_content="Pay in 30 days.", metadata={"source": "0.pdf"})
    ]
    search.assert_called_once_with("terms?", k=10)


def test_contextual_empty_results_skip_llm(monkeypatch):
    monkeypatch.setattr(retrieval, "_similarity", lambda *a, **kw: [])
    llm = Mock()
    monkeypatch.setattr(retrieval, "get_llm", llm)
    assert retrieval._contextual("question") == []
    llm.assert_not_called()


def test_ask_llm_runs_retrieval_prompt_model_and_parser(monkeypatch):
    store = Mock()
    store.similarity_search.return_value = [
        Document(page_content="Pay in 30 days.", metadata={"source": "terms.pdf"})
    ]
    monkeypatch.setattr(retrieval, "get_vector_store", lambda: store)

    def answer(prompt):
        messages = prompt.to_messages()
        assert "[terms.pdf]\nPay in 30 days." in messages[0].content
        assert messages[-1].content == "Payment terms?"
        return AIMessage(content="Payment is due in 30 days [terms.pdf].")

    monkeypatch.setattr(retrieval, "get_llm", lambda: RunnableLambda(answer))
    log = Mock()
    monkeypatch.setattr(retrieval, "log", log)
    assert retrieval.ask_llm("Payment terms?") == "Payment is due in 30 days [terms.pdf]."
    store.similarity_search.assert_called_once_with("Payment terms?", k=3)
    assert log.info.call_count == 3


def test_ask_llm_logs_and_propagates_failure(monkeypatch):
    chain = Mock()
    chain.invoke.side_effect = RuntimeError("Model unavailable")
    builder = Mock(return_value=chain)
    monkeypatch.setattr(retrieval, "build_rag_chain", builder)
    log = Mock()
    monkeypatch.setattr(retrieval, "log", log)
    with pytest.raises(RuntimeError, match="Model unavailable"):
        retrieval.ask_llm("question", "multiquery")
    builder.assert_called_once_with("multiquery")
    log.exception.assert_called_once()
