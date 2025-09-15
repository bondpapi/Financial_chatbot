"""
Centralized config & API key management.
- Loads .env
- Exposes get_setting(), require_key()
- Optional LangSmith tracing
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=False)

def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)

def set_setting(name: str, value: str) -> None:
    os.environ[name] = value

def require_key(name: str) -> str:
    v = os.getenv(name, "")
    if not v:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it in .env or your deployment environment."
        )
    return v

def enable_langsmith_if_configured(logger=None) -> None:
    """
    Enable LangSmith tracing if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY are set.
    """
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {"1", "true", "yes"}
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if tracing and api_key:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        if logger:
            logger.info("LangSmith tracing enabled.")
    elif logger:
        logger.info("LangSmith tracing not enabled (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY).")
