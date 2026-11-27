"""Работа с моделью: загрузка чекпоинта, предобработка и предсказание."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import models, transforms

from app.config import CLASS_NAMES_PATH, MODEL_PATH
from app.logging_setup import log

IMG_SIZE = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_VERSION = "resnet18-finetuned-v1"

_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def preprocess(image: Image.Image) -> torch.Tensor:
    """Предобработка одной картинки -> тензор [1, 3, H, W]."""
    return _transform(image.convert("RGB")).unsqueeze(0)


def _load_class_names(model_path: Path, n_classes: int) -> List[str]:
    """Названия классов: class_names.json рядом с моделью (или CLASS_NAMES_PATH).

    Порядок должен совпадать с порядком классов при обучении
    (например, train_dataset.classes из ImageFolder).
    """
    path = CLASS_NAMES_PATH or model_path.with_name("class_names.json")
    if path.exists():
        names = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(names, dict):  # поддержка {"0": "beagle", ...}
            names = [names[str(i)] for i in range(len(names))]
        names = [str(n) for n in names]
        if len(names) != n_classes:
            raise ValueError(
                f"в {path.name} {len(names)} классов, а в модели {n_classes}"
            )
        return names
    log.warning(
        "файл с названиями классов не найден, используются class_0..class_N",
        extra={"event": "class_names_missing"},
    )
    return [f"class_{i}" for i in range(n_classes)]


def load_model(
    path: Path = MODEL_PATH, device: str = "cpu"
) -> Tuple[torch.nn.Module, List[str]]:
    """Загружает веса ResNet18.

    Поддерживаются два формата файла:
      1) обычный state_dict (как у вас), голова fc = Sequential(Dropout, Linear)
         или одиночный Linear (определяется по ключам);
      2) словарь {"class_names": [...], "model_state": state_dict}.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"чекпоинт не найден: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    names_from_ckpt = None
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        names_from_ckpt = list(ckpt["class_names"])
        state = ckpt["model_state"]
    else:
        state = ckpt

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    if "fc.1.weight" in state:  # fc = Sequential(Dropout, Linear)
        n_classes = state["fc.1.weight"].shape[0]
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.2), torch.nn.Linear(in_features, n_classes)
        )
    elif "fc.weight" in state:  # fc = Linear
        n_classes = state["fc.weight"].shape[0]
        model.fc = torch.nn.Linear(in_features, n_classes)
    else:
        raise KeyError("в весах нет ни fc.weight, ни fc.1.weight")

    model.load_state_dict(state)
    model.to(device).eval()

    class_names = names_from_ckpt or _load_class_names(path, n_classes)
    return model, class_names


def predict_top_k(
    model: torch.nn.Module,
    class_names: List[str],
    image: Image.Image,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    top_k = max(1, min(int(top_k), len(class_names)))
    tensor = preprocess(image)

    with torch.inference_mode():
        probs = torch.softmax(model(tensor)[0], dim=0)

    values, indices = torch.topk(probs, k=top_k)
    return [
        (class_names[int(i)], round(float(v), 4)) for v, i in zip(values, indices)
    ]
