# financial_bot/guardrails.py
import re
from typing import Tuple

# Very light PII redaction for logs/UI
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CREDIT = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

INJECTION_MARKERS = (
    "ignore previous", "disregard previous", "act as system", "you are no longer",
    "reset instructions", "developer mode", "strip safety", "override guard", "tool output:", "function schema:"
)

PROHIBITED_FINANCE = (
    "insider info", "non-public info", "front run", "pump and dump", "guarantee profits"
)

ADVICE_TRIGGERS = (
    "should i buy", "should i sell", "what should i invest", "best stock to buy",
    "which stock to buy", "allocate my portfolio", "rebalance my portfolio"
)

DISCLAIMER = (
    "> **Not financial advice.** Educational information only. "
    "Markets carry risk; Do your own research or Consult a licensed professional."
)

def redact_sensitive(text: str) -> str:
    if not text:
        return text
    text = _EMAIL.sub("[redacted-email]", text)
    text = _CREDIT.sub("[redacted-card]", text)
    text = _SSN.sub("[redacted-ssn]", text)
    return text

def looks_like_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in INJECTION_MARKERS)

def contains_prohibited_finance(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in PROHIBITED_FINANCE)

def needs_disclaimer(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ADVICE_TRIGGERS)

def apply_guardrails(user_query: str) -> Tuple[bool, str]:
    """
    Returns (ok, message_or_query). If ok=False, message is a refusal string.
    If ok=True, message is the (possibly sanitized) query string.
    """
    if contains_prohibited_finance(user_query):
        return False, (
            "❌ I can’t help with requests involving insider information or illegal activity. "
            "I can, however, explain how markets work or discuss public information."
        )
    if looks_like_injection(user_query):
        # Neutralize typical injection cues (soft approach)
        cleaned = user_query
        for marker in INJECTION_MARKERS:
            cleaned = cleaned.replace(marker, "[removed]")
        return True, cleaned
    return True, user_query
