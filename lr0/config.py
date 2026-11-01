"""Загрузка и проверка конфигурации запуска."""
import copy
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Конфигурация некорректна: запуск не начинается."""


# Значения по умолчанию. Файл конфигурации и аргументы командной строки
# переопределяют их; итоговый словарь сохраняется в каталог запуска.
DEFAULTS = {
    "experiment": "lr0",
    "series": "debug",
    "split_seed": 0,
    "seed": 0,
    "model": {"n_layers": 1, "hidden": 64, "dropout": 0.0},
    "train": {"lr": 0.001, "weight_decay": 0.0, "batch_size": 64, "epochs": 30},
    "runtime": {"device": "cpu", "num_threads": 1},
}


def deep_merge(base, override):
    """Рекурсивно накладывает override на base, не меняя исходные словари."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(cfg):
    """Проверяет конфигурацию; при ошибке бросает ConfigError с понятным текстом."""
    errors = []
    m, t, r = cfg.get("model", {}), cfg.get("train", {}), cfg.get("runtime", {})

    for key in ("seed", "split_seed"):
        if not _is_int(cfg.get(key)) or cfg[key] < 0:
            errors.append(f"{key} должен быть целым >= 0, получено {cfg.get(key)!r}")

    if not _is_int(m.get("n_layers")) or m["n_layers"] < 1:
        errors.append(f"model.n_layers должен быть целым >= 1, получено {m.get('n_layers')!r}")
    if not _is_int(m.get("hidden")) or m["hidden"] < 1:
        errors.append(f"model.hidden должен быть целым >= 1, получено {m.get('hidden')!r}")
    if not _is_num(m.get("dropout")) or not 0 <= m["dropout"] < 1:
        errors.append(f"model.dropout должен быть в [0, 1), получено {m.get('dropout')!r}")

    if not _is_num(t.get("lr")) or t["lr"] <= 0:
        errors.append(f"train.lr должен быть > 0, получено {t.get('lr')!r}")
    if not _is_num(t.get("weight_decay")) or t["weight_decay"] < 0:
        errors.append(f"train.weight_decay должен быть >= 0, получено {t.get('weight_decay')!r}")
    if not _is_int(t.get("batch_size")) or t["batch_size"] < 1:
        errors.append(f"train.batch_size должен быть целым >= 1, получено {t.get('batch_size')!r}")
    if not _is_int(t.get("epochs")) or t["epochs"] < 1:
        errors.append(f"train.epochs должен быть целым >= 1, получено {t.get('epochs')!r}")

    if r.get("device") != "cpu":
        errors.append(f"runtime.device: поддерживается только 'cpu', получено {r.get('device')!r}")
    if not _is_int(r.get("num_threads")) or r["num_threads"] < 1:
        errors.append(f"runtime.num_threads должен быть целым >= 1, получено {r.get('num_threads')!r}")

    if errors:
        raise ConfigError("Некорректная конфигурация:\n  - " + "\n  - ".join(errors))
    return cfg


def load_config(path, overrides=None):
    """Читает YAML, накладывает на значения по умолчанию и overrides, проверяет."""
    with open(Path(path), encoding="utf-8") as f:
        from_file = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, from_file)
    cfg = deep_merge(cfg, overrides or {})
    return validate(cfg)
