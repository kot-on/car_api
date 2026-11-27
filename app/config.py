"""Настройки сервиса: всё берётся из переменных окружения."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

API_VERSION = "1.1.0"

JWT_SECRET_FROM_ENV = bool(os.getenv("JWT_SECRET"))
JWT_SECRET = os.getenv("JWT_SECRET") or secrets.token_urlsafe(32)
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR.parent / "resnet18-finetuned-v1.pt")))
_cn = os.getenv("CLASS_NAMES_PATH")
CLASS_NAMES_PATH = Path(_cn) if _cn else None  
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))


API_USER = os.getenv("API_USER", "student")
API_PASSWORD = os.getenv("API_PASSWORD", "changeme")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "30"))

JWT_SECRET_FROM_ENV = bool(os.getenv("JWT_SECRET"))
JWT_SECRET = os.getenv("JWT_SECRET") or secrets.token_urlsafe(32)
