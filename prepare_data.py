"""Готовит data/split.json и data/manifest.json из load_digits (scikit-learn).

    python prepare_data.py

Если split.json уже есть, он не перезаписывается: скрипт только проверяет,
что заново посчитанное разбиение с ним совпадает.
"""
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from sklearn.datasets import load_digits  # noqa: E402

from lr0.data import MANIFEST_PATH, SPLIT_PATH, check_split, make_split, sha256_file  # noqa: E402

SPLIT_SEED = 0


def main():
    X, y = load_digits(return_X_y=True)
    split = make_split(y, SPLIT_SEED)
    check_split(split, y)
    text = json.dumps(split, indent=1) + "\n"

    if SPLIT_PATH.exists():
        if SPLIT_PATH.read_text(encoding="utf-8") != text:
            sys.exit("data/split.json отличается от заново посчитанного разбиения — проверьте версию scikit-learn")
        print("data/split.json уже есть и совпадает с разбиением")
    else:
        SPLIT_PATH.parent.mkdir(exist_ok=True)
        SPLIT_PATH.write_text(text, encoding="utf-8")
        print("Записан data/split.json")

    sizes = {p: len(split[p]) for p in ("train", "val", "test")}
    manifest = {
        "datasets": {
            "digits": {
                "source": "sklearn.datasets.load_digits (копия UCI Optical Recognition of Handwritten Digits, тестовая часть)",
                "version": f"scikit-learn=={version('scikit-learn')}",
                "records_total": int(len(y)),
                "records_used": int(len(y)),
                "excluded": 0,
                "excluded_reason": "нет",
                "features": "64 признака (пиксели 8x8, значения 0..16), при загрузке делятся на 16",
                "labels": "цифры 0..9",
                "split_rule": (
                    "стратифицированно по классам: train_test_split(test_size=0.2, random_state=split_seed) "
                    "отделяет test, затем train_test_split(test_size=0.25, random_state=split_seed) "
                    "от остатка отделяет val"
                ),
                "split_seed": SPLIT_SEED,
                "split_sizes": sizes,
                "files": {
                    "data/split.json": {
                        "sha256": sha256_file(SPLIT_PATH),
                        "records": sum(sizes.values()),
                    }
                },
            }
        }
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Записан data/manifest.json; размеры частей: {sizes}")


if __name__ == "__main__":
    main()
