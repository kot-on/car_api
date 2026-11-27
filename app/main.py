"""FastAPI-сервис классификации пород собак.
Запуск: python -m app.main   или   uvicorn app.main:app --reload
"""
from __future__ import annotations

import io
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app import config
from app.auth import authenticate, create_token, require_user
from app.errors import register_error_handlers
from app.logging_setup import log
from app.model import MODEL_VERSION, load_model, predict_top_k
from app.ratelimit import limit_by_ip

state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.JWT_SECRET_FROM_ENV:
        log.warning("JWT_SECRET не задан: используется случайный секрет",
                    extra={"event": "config"})
    if config.MODEL_PATH.exists():
        try:
            state["model"], state["class_names"] = load_model(config.MODEL_PATH)
            state["version"] = MODEL_VERSION
            log.info("модель загружена",
                     extra={"event": "model_loaded", "model_version": MODEL_VERSION})
        except Exception:
            log.error("не удалось загрузить модель", exc_info=True,
                      extra={"event": "model_load_failed"})
    else:
        log.error("файл модели не найден", extra={"event": "model_missing"})
    yield
    state.clear()


app = FastAPI(title="Car Api", version=config.API_VERSION, lifespan=lifespan)
register_error_handlers(app)


# ---------- middleware: request_id + структурированный лог запроса ----------
@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        log.info(
            "запрос",
            extra={
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                "client_ip": request.client.host if request.client else None,
            },
        )


# ---------- схемы ----------
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class Prediction(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    top_k: list[Prediction]
    model_version: str
    elapsed_ms: float


def _require_model() -> None:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="модель не загружена")


# ---------- эндпоинты ----------
@app.get("/health")
def health() -> Dict[str, Any]:
    """Открытый эндпоинт: сервис жив, модель загружена или нет."""
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/version")
def version() -> Dict[str, str]:
    return {"api_version": config.API_VERSION, "model_version": MODEL_VERSION}


@app.post("/auth/token", response_model=TokenResponse)
def token(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    limit_by_ip(request)
    if not authenticate(form.username, form.password):
        log.warning("неудачная попытка входа", extra={"event": "login_failed"})
        raise HTTPException(
            status_code=401,
            detail="неверный логин или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token, expires_in = create_token(form.username)
    log.info("успешный вход", extra={"event": "login_ok", "user": form.username})
    return TokenResponse(access_token=access_token, expires_in=expires_in)


@app.get("/classes")
def classes(user: str = Depends(require_user)) -> Dict[str, Any]:
    _require_model()
    return {"classes": state["class_names"], "count": len(state["class_names"])}


@app.post("/predict", response_model=PredictResponse)
def predict(
    file: UploadFile = File(...),
    top_k: int = 3,
    user: str = Depends(require_user),
) -> PredictResponse:
    start = time.perf_counter()
    _require_model()

    n_classes = len(state["class_names"])
    if not 1 <= top_k <= n_classes:
        raise HTTPException(status_code=422, detail=f"top_k должен быть от 1 до {n_classes}")

    # читаем не больше лимита + 1 байт, чтобы не грузить огромный файл в память
    data = file.file.read(config.MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="пустой файл")
    if len(data) > config.MAX_BYTES:
        raise HTTPException(
            status_code=413, detail=f"файл больше {config.MAX_UPLOAD_MB} МБ"
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # принудительно декодируем, чтобы поймать битые файлы здесь
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="не удалось прочитать изображение")

    top = predict_top_k(state["model"], state["class_names"], image, top_k)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    log.info(
        "предсказание",
        extra={"event": "predict", "user": user, "top_k": top_k,
               "model_version": state["version"], "elapsed_ms": elapsed_ms},
    )
    return PredictResponse(
        predicted_label=top[0][0],
        confidence=top[0][1],
        top_k=[Prediction(label=n, probability=p) for n, p in top],
        model_version=state["version"],
        elapsed_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT,
                reload=False, log_level=config.LOG_LEVEL.lower())
