"""JWT-авторизация: POST /auth/token выдаёт токен, остальные эндпоинты проверяют его."""
from __future__ import annotations

import secrets
import time

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import config
from app.ratelimit import limit_by_user

bearer = HTTPBearer(auto_error=False)


def _same(a: str, b: str) -> bool:
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def authenticate(username: str, password: str) -> bool:
    # обе проверки выполняются всегда, чтобы не выдавать, что именно неверно
    user_ok = _same(username, config.API_USER)
    pass_ok = _same(password, config.API_PASSWORD)
    return user_ok and pass_ok


def create_token(username: str) -> tuple[str, int]:
    expires_in = config.JWT_EXPIRE_MIN * 60
    now = int(time.time())
    payload = {"sub": username, "iat": now, "exp": now + expires_in}
    token = jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)
    return token, expires_in


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=message,
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str:
    if creds is None:
        raise _unauthorized("нужен токен: Authorization: Bearer <token>")
    try:
        payload = jwt.decode(
            creds.credentials, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("срок действия токена истёк")
    except jwt.InvalidTokenError:
        raise _unauthorized("неверный токен")
    user = payload.get("sub")
    if not user:
        raise _unauthorized("неверный токен")
    limit_by_user(user)  # 429, если превышен лимит
    return user
