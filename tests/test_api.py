import io
import time

import jwt
from PIL import Image

from app import config, ratelimit


def png_bytes(size=(64, 64), color=(120, 80, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def post_predict(client, auth, data, top_k=None, name="dog.png"):
    params = {"top_k": top_k} if top_k is not None else {}
    return client.post("/predict", headers=auth, params=params,
                       files={"file": (name, data, "image/png")})


def assert_error(resp, code):
    assert resp.status_code == code
    body = resp.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


# ----- открытые эндпоинты -----
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json()["api_version"] == config.API_VERSION
    assert r.json()["model_version"]


# ----- авторизация -----
def test_token_ok(client):
    r = client.post("/auth/token", data={"username": "tester", "password": "secret"})
    assert r.status_code == 200
    assert r.json()["token_type"] == "bearer"


def test_token_wrong_password_401(client):
    r = client.post("/auth/token", data={"username": "tester", "password": "bad"})
    assert_error(r, 401)


def test_no_token_401(client):
    assert_error(client.get("/classes"), 401)
    r = client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    assert_error(r, 401)


def test_bad_token_401(client):
    r = client.get("/classes", headers={"Authorization": "Bearer abc.def.ghi"})
    assert_error(r, 401)


def test_expired_token_401(client):
    old = jwt.encode({"sub": "tester", "exp": int(time.time()) - 10},
                     config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)
    r = client.get("/classes", headers={"Authorization": f"Bearer {old}"})
    assert_error(r, 401)


# ----- /classes -----
def test_classes(client, auth):
    r = client.get("/classes", headers=auth)
    assert r.status_code == 200
    assert r.json() == {"classes": ["beagle", "boxer", "pug"], "count": 3}


# ----- /predict -----
def test_predict_ok(client, auth):
    r = post_predict(client, auth, png_bytes(), top_k=2)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_label"] == body["top_k"][0]["label"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 2
    probs = [p["probability"] for p in body["top_k"]]
    assert probs == sorted(probs, reverse=True)
    assert body["model_version"]
    assert "elapsed_ms" in body


def test_predict_default_top_k(client, auth):
    assert len(post_predict(client, auth, png_bytes()).json()["top_k"]) == 3


def test_predict_empty_file_400(client, auth):
    assert_error(post_predict(client, auth, b""), 400)


def test_predict_not_image_400(client, auth):
    assert_error(post_predict(client, auth, b"this is not an image"), 400)


def test_predict_truncated_image_400(client, auth):
    big = io.BytesIO()
    Image.effect_noise((200, 200), 80).convert("RGB").save(big, format="JPEG")
    assert_error(post_predict(client, auth, big.getvalue()[:300]), 400)


def test_predict_too_large_413(client, auth):
    assert_error(post_predict(client, auth, b"\0" * (config.MAX_BYTES + 10)), 413)


def test_predict_top_k_zero_422(client, auth):
    assert_error(post_predict(client, auth, png_bytes(), top_k=0), 422)


def test_predict_top_k_too_big_422(client, auth):
    assert_error(post_predict(client, auth, png_bytes(), top_k=99), 422)


def test_predict_top_k_not_int_422(client, auth):
    assert_error(post_predict(client, auth, png_bytes(), top_k="abc"), 422)


def test_predict_no_file_422(client, auth):
    assert_error(client.post("/predict", headers=auth), 422)


# ----- 429 и 503 -----
def test_rate_limit_429(client, auth, monkeypatch):
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MIN", 2)
    ratelimit.reset()
    codes = [client.get("/classes", headers=auth).status_code for _ in range(4)]
    assert codes[:2] == [200, 200]
    assert codes[-1] == 429
    r = client.get("/classes", headers=auth)
    assert_error(r, 429)
    assert "retry-after" in r.headers


def test_model_not_loaded_503(client, auth, monkeypatch):
    from app.main import state
    monkeypatch.delitem(state, "model")
    assert_error(client.get("/classes", headers=auth), 503)
    assert_error(post_predict(client, auth, png_bytes()), 503)
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
