"""
User input validation & sanitation.
"""
import re

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

def clean_text(s: str, max_len: int = 2000) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s[:max_len]

def validate_query(q: str) -> str:
    q = clean_text(q, 2000)
    if not q or len(q) < 3:
        raise ValueError("Query is too short. Please ask a more specific question.")
    return q

def normalize_ticker(t: str) -> str:
    t = (t or "").upper().strip()
    t = t.replace(" ", "")
    if not _TICKER_RE.match(t):
        raise ValueError("Invalid ticker format. Use letters/numbers like AAPL, MSFT, BRK.B, etc.")
    return t

