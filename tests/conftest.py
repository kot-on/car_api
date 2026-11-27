"""Общие фикстуры. Для тестов создаётся маленький чекпоинт со случайными весами."""
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from torchvision import models

CLASSES = ["beagle", "boxer", "pug"]
_tmp = Path(tempfile.mkdtemp())
CKPT = _tmp / "test_model.pt"

# переменные окружения нужно задать ДО импорта app
os.environ["MODEL_PATH"] = str(CKPT)
os.environ["API_USER"] = "tester"
os.environ["API_PASSWORD"] = "secret"
os.environ["JWT_SECRET"] = "test-secret-test-secret-test-secret"
os.environ["MAX_UPLOAD_MB"] = "1"
os.environ["RATE_LIMIT_PER_MIN"] = "1000"

# формат как у реального файла: обычный state_dict, fc = Sequential(Dropout, Linear),
# названия классов лежат рядом в class_names.json
_m = models.resnet18(weights=None)
_m.fc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(_m.fc.in_features, len(CLASSES)))
torch.save(_m.state_dict(), CKPT)
(_tmp / "class_names.json").write_text(json.dumps(CLASSES), encoding="utf-8")

from fastapi.testclient import TestClient  # noqa: E402

from app import ratelimit  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_limits(monkeypatch):
    ratelimit.reset()
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MIN", 1000)


@pytest.fixture()
def client():
    with TestClient(app) as c:  # with -> запускается lifespan (загрузка модели)
        yield c


@pytest.fixture()
def auth(client):
    r = client.post("/auth/token", data={"username": "tester", "password": "secret"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
