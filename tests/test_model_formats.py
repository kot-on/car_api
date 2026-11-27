import json

import torch
from torchvision import models

from app.model import load_model


def _head(n, seq):
    m = models.resnet18(weights=None)
    lin = torch.nn.Linear(m.fc.in_features, n)
    m.fc = torch.nn.Sequential(torch.nn.Dropout(0.2), lin) if seq else lin
    return m


def test_plain_state_dict_with_names_file(tmp_path):
    torch.save(_head(4, seq=True).state_dict(), tmp_path / "m.pt")
    (tmp_path / "class_names.json").write_text(json.dumps(["a", "b", "c", "d"]))
    model, names = load_model(tmp_path / "m.pt")
    assert names == ["a", "b", "c", "d"]
    assert model(torch.zeros(1, 3, 160, 160)).shape == (1, 4)


def test_missing_names_file_falls_back(tmp_path):
    torch.save(_head(3, seq=True).state_dict(), tmp_path / "m.pt")
    _, names = load_model(tmp_path / "m.pt")
    assert names == ["class_0", "class_1", "class_2"]


def test_names_count_mismatch_raises(tmp_path):
    torch.save(_head(3, seq=True).state_dict(), tmp_path / "m.pt")
    (tmp_path / "class_names.json").write_text(json.dumps(["a", "b"]))
    try:
        load_model(tmp_path / "m.pt")
    except ValueError:
        return
    raise AssertionError("ожидалась ValueError")


def test_plain_linear_head_and_legacy_dict(tmp_path):
    torch.save(_head(2, seq=False).state_dict(), tmp_path / "lin.pt")
    (tmp_path / "class_names.json").write_text(json.dumps(["x", "y"]))
    assert load_model(tmp_path / "lin.pt")[1] == ["x", "y"]

    torch.save({"class_names": ["p", "q"], "model_state": _head(2, seq=False).state_dict()},
               tmp_path / "old.pt")
    assert load_model(tmp_path / "old.pt")[1] == ["p", "q"]
