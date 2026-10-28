"""Shared, deterministic checks for known prompt overrides.

This is a conservative heuristic: it can reject quoted examples and cannot identify
all indirect, encoded, or multilingual attacks. Keep instruction/data separation too.
"""
import re
import unicodedata

PROMPT_INJECTION_REFUSAL = (
    "Sorry, I can’t proceed because the request or supporting content "
    "appears to ask me to override my instructions. "
    "I’m Meridian AI, your procurement audit assistant. "
    "Please rephrase your procurement question or review the supporting content."
)

_PATTERNS = (
    re.compile(r"\[\s*(?:system|developer)\s*\]|<\|(?:im_start|start_header_id)\|>\s*(?:system|developer)", re.I),
    re.compile(r"\b(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your)\s+)*(?:previous|prior|above|system|developer)\s+(?:instructions?|prompts?|rules?|messages?)", re.I),
)


def contains_prompt_injection(text: str) -> bool:
    normalized = unicodedata.normalize('NFKC', text)
    return any(pattern.search(normalized) for pattern in _PATTERNS)
