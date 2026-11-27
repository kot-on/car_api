"""Структурированные (JSON) логи. В логи не попадают файлы, пароли и токены."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.config import LOG_LEVEL

# Поля, которые можно передавать через extra={...}
_EXTRA_FIELDS = (
    "request_id", "method", "path", "status", "elapsed_ms",
    "user", "event", "model_version", "top_k", "client_ip",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                data[key] = value
        if record.exc_info:
            data["exc_type"] = record.exc_info[0].__name__
        return json.dumps(data, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("dogapi")
    logger.handlers = [handler]
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    return logger


log = setup_logging()
