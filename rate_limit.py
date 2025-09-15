"""
Simple in-process rate limiting decorator (token-bucket-ish).
For multi-instance deployments, switch to Redis-based limiter.
"""
import time
import threading
from collections import deque
from functools import wraps

_lock = threading.Lock()
_calls = {}

def rate_limited(calls: int = 10, period: int = 60, key: str = None):
    """
    Allow at most `calls` per `period` seconds for a given key (function or explicit key).
    """
    def decorator(fn):
        bucket_key = key or fn.__name__
        _calls.setdefault(bucket_key, deque())

        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = time.time()
            with _lock:
                dq = _calls[bucket_key]
                while dq and (now - dq[0]) > period:
                    dq.popleft()
                if len(dq) >= calls:
                    # Backoff: sleep until the earliest call expires
                    sleep_for = period - (now - dq[0])
                    time.sleep(max(sleep_for, 0))
                    # After sleep, re-check window
                    return wrapper(*args, **kwargs)
                dq.append(now)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

