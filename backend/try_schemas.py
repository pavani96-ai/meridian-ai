"""Offline API schema checks. Run from backend/: uv run python try_schemas.py."""
from pydantic import ValidationError

from api.schemas import AuditRequest, AuditResponse, QueryRequest, QueryResponse


def expect_invalid(model, payload):
    try:
        model.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError(f'{model.__name__} accepted invalid input: {payload!r}')


def main():
    audit = AuditRequest(request_text='  Review invoice.  ')
    assert audit.request_text == 'Review invoice.'
    assert AuditRequest.model_validate_json(audit.model_dump_json()) == audit
    for payload in ({}, {'request_text': ''}, {'request_text': '  '}, {'request_text': None}):
        expect_invalid(AuditRequest, payload)
    print('PASS: AuditRequest valid input, trimming, and invalid input rejection')

    assert QueryRequest(query=' Question ').query == 'Question'
    assert QueryRequest(query='Question').retriever_type == 'similarity'
    for strategy in ('similarity', 'multiquery', 'contextual'):
        assert QueryRequest(query='Question', retriever_type=strategy).retriever_type == strategy
    for payload in ({}, {'query': ''}, {'query': '  '}, {'query': None},
                    {'query': 'Question', 'retriever_type': 'unsupported'}):
        expect_invalid(QueryRequest, payload)
    print('PASS: QueryRequest validation, default, and supported retrievers')

    payload = dict(risk_result='risk', tax_result='tax', control_result='control', cfo_memo='memo')
    response = AuditResponse(**payload)
    assert response.model_dump() == payload
    assert AuditResponse.model_validate_json(response.model_dump_json()) == response
    for field in payload:
        expect_invalid(AuditResponse, {k: v for k, v in payload.items() if k != field})
    print('PASS: AuditResponse required fields and JSON round-trip')

    response = QueryResponse(answer='Policy answer')
    assert QueryResponse.model_validate_json(response.model_dump_json()) == response
    expect_invalid(QueryResponse, {})
    expect_invalid(QueryResponse, {'answer': None})
    print('PASS: QueryResponse validation and JSON round-trip')
    print('\nAll schema checks passed. No API calls were made.')


if __name__ == '__main__':
    main()
