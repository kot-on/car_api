"""Простой лимит запросов: скользящее окно 60 секунд, в памяти процесса."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

from app.config import RATE_LIMIT_PER_MIN

WINDOW_SEC = 60.0
_hits: dict[str, deque] = defaultdict(deque)
_lock = Lock()


def _check(key: str) -> None:
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.monotonic()
    with _lock:
        q = _hits[key]
        while q and now - q[0] > WINDOW_SEC:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            retry = max(1, int(WINDOW_SEC - (now - q[0])))
            raise HTTPException(
                status_code=429,
                detail=f"слишком много запросов, повторите через {retry} с",
                headers={"Retry-After": str(retry)},
            )
        q.append(now)


def limit_by_ip(request: Request) -> None:
    """Для /auth/token: защита от перебора пароля."""
    _check("ip:" + (request.client.host if request.client else "unknown"))


def limit_by_user(user: str) -> None:
    _check("user:" + user)


def reset() -> None:
    with _lock:
        _hits.clear()
