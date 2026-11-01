"""Данные: разбиение load_digits, хеши, загрузка с проверками."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SPLIT_PATH = Path("data/split.json")
MANIFEST_PATH = Path("data/manifest.json")
PARTS = ("train", "val", "test")
N_CLASSES = 10


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def make_split(y, split_seed=0):
    """Стратифицированное разбиение: test 20%, затем val 25% остатка (1077/360/360)."""
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=split_seed)
    train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=split_seed)
    return {
        "split_seed": split_seed,
        "train": sorted(int(i) for i in train),
        "val": sorted(int(i) for i in val),
        "test": sorted(int(i) for i in test),
    }


def check_split(split, y):
    """Части не пересекаются, покрывают весь набор и содержат все классы."""
    n = len(y)
    for part in PARTS:
        if part not in split:
            raise ValueError(f"В split.json нет части '{part}'")
        ids = split[part]
        if len(ids) == 0:
            raise ValueError(f"Часть '{part}' пустая")
        if len(set(ids)) != len(ids):
            raise ValueError(f"В части '{part}' есть повторяющиеся индексы")
        if min(ids) < 0 or max(ids) >= n:
            raise ValueError(f"В части '{part}' индексы вне диапазона 0..{n - 1}")
        classes = set(int(c) for c in y[ids])
        if len(classes) != N_CLASSES:
            raise ValueError(f"В части '{part}' только {len(classes)} классов из {N_CLASSES}")
    sets = [set(split[p]) for p in PARTS]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Части разбиения пересекаются")
    if sum(len(s) for s in sets) != n:
        raise ValueError(f"Части в сумме дают {sum(len(s) for s in sets)}, а не {n}")


def load_data(split_path=SPLIT_PATH, manifest_path=MANIFEST_PATH):
    """Загружает данные и разбиение.

    Хеш считается по тем же байтам, из которых читается разбиение, и сверяется
    с манифестом. Признаки делятся на 16 (масштаб известен заранее).
    """
    raw = Path(split_path).read_bytes()
    split_sha = sha256_bytes(raw)
    split = json.loads(raw.decode("utf-8"))

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = manifest["datasets"]["digits"]["files"]["data/split.json"]["sha256"]
    if split_sha != expected:
        raise ValueError(f"SHA-256 split.json ({split_sha}) не совпадает с манифестом ({expected})")

    X, y = load_digits(return_X_y=True)
    check_split(split, y)
    X = (X / 16.0).astype(np.float32)
    y = y.astype(np.int64)
    return {
        "X": X,
        "y": y,
        "split": {p: np.array(split[p], dtype=np.int64) for p in PARTS},
        "split_sha256": split_sha,
    }
