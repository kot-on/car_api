# Dog Breed API

REST-сервис классификации пород собак (FastAPI + PyTorch, ResNet18).
Модель загружается один раз при старте, на запросе выполняется только инференс.

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Положите чекпоинт `resnet18_finetuned.pt` в корень проекта (рядом с папкой `app`)
либо укажите путь в `MODEL_PATH`. Поддерживается обычный `state_dict` ResNet18 (голова `fc` — `Linear` или `Sequential(Dropout, Linear)`).
Названия классов положите рядом в `class_names.json` (JSON-список в порядке обучения, например `["beagle", "boxer", ...]`) или укажите путь в `CLASS_NAMES_PATH`.

Задайте переменные окружения (см. `.env.example`) и запустите:

```powershell
$env:JWT_SECRET = "длинная_случайная_строка"
$env:API_PASSWORD = "мой_пароль"
python -m app.main
```

Документация Swagger: http://127.0.0.1:8000/docs

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `MODEL_PATH` | `../resnet18_finetuned.pt` | путь к чекпоинту |
| `CLASS_NAMES_PATH` | `class_names.json` рядом с моделью | список названий классов |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | адрес и порт |
| `LOG_LEVEL` | `INFO` | уровень логов |
| `MAX_UPLOAD_MB` | `5` | максимальный размер файла |
| `RATE_LIMIT_PER_MIN` | `30` | запросов в минуту на пользователя (0 = выкл.) |
| `API_USER` / `API_PASSWORD` | `student` / `changeme` | логин и пароль для получения токена |
| `JWT_SECRET` | случайный при старте | секрет подписи JWT |
| `JWT_EXPIRE_MIN` | `30` | срок жизни токена, минуты |

## Эндпоинты

| Метод | Путь | Авторизация | Описание |
|---|---|---|---|
| GET | `/health` | нет | `{"status": "ok", "model_loaded": true}` |
| GET | `/version` | нет | версии API и модели |
| POST | `/auth/token` | нет | логин и пароль (form-data) -> JWT |
| GET | `/classes` | Bearer | список классов |
| POST | `/predict` | Bearer | картинка (`file`) и `top_k` -> предсказание |

## Примеры

```bash
# получить токен
curl -X POST http://127.0.0.1:8000/auth/token -d "username=student&password=changeme"

# предсказание
curl -X POST "http://127.0.0.1:8000/predict?top_k=3" \
  -H "Authorization: Bearer <TOKEN>" -F "file=@dog.jpg"
```

Ответ `/predict`:

```json
{
  "predicted_label": "beagle",
  "confidence": 0.8731,
  "top_k": [{"label": "beagle", "probability": 0.8731}, {"label": "boxer", "probability": 0.0912}],
  "model_version": "resnet18-finetuned-v1",
  "elapsed_ms": 41.2
}
```

## Ошибки

Формат всех ошибок одинаков: `{"error": {"code": 422, "message": "..."}}`

| Код | Причина |
|---|---|
| 400 | пустой или битый файл |
| 401 | нет, неверный или просроченный токен; неверный логин/пароль |
| 413 | файл больше `MAX_UPLOAD_MB` |
| 422 | неверный `top_k` или отсутствует `file` |
| 429 | превышен лимит запросов (заголовок `Retry-After`) |
| 503 | модель не загружена |

## Логи

JSON, по строке на событие: время, уровень, `request_id`, метод, путь, статус, время обработки.
Содержимое файлов, пароли и токены не логируются.

## Тесты

```bash
pytest -q
```

Тесты сами создают маленький чекпоинт со случайными весами, настоящая модель не нужна.

## Docker

```bash
docker build -t dog-api .
docker run -p 8000:8000 -e JWT_SECRET=... -e API_PASSWORD=... -v ./resnet18_finetuned.pt:/srv/resnet18_finetuned.pt dog-api
```

## Важно

Параметры предобработки в `app/model.py` (`IMG_SIZE`, `MEAN`, `STD`) должны совпадать с обучением.
