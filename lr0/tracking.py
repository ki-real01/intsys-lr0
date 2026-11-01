"""Git, метаданные, каталог запуска и журнал MLflow."""
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

TRACKING_URI = "sqlite:///mlflow.db"
HISTORY_FIELDS = ["epoch", "train_loss", "val_loss", "val_acc", "epoch_time"]
PRED_FIELDS = ["split", "index", "y_true", "y_pred"]


# ---------- git ----------

def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def git_commit():
    return _git("rev-parse", "HEAD")


def git_dirty():
    """Изменённые и новые неигнорируемые файлы (пусто — дерево чистое)."""
    return _git("status", "--porcelain")


def require_clean_tree():
    """Итоговый запуск только из закоммиченного кода."""
    dirty = git_dirty()
    if dirty:
        raise SystemExit(
            "Есть незакоммиченные изменения — итоговый запуск невозможен:\n" + dirty
            + "\n(для отладки добавьте --debug)"
        )
    return git_commit()


# ---------- окружение ----------

def library_versions():
    libs = ["torch", "numpy", "scikit-learn", "optuna", "mlflow", "pyyaml", "pandas"]
    return {"python": platform.python_version(), **{lib: version(lib) for lib in libs}}


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------- каталог запуска ----------

class RunDir:
    """runs/<run_id>/: config.yaml, metadata.json, history.csv, predictions.csv, metrics.json."""

    def __init__(self, root, run_id):
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {}

    def write_config(self, cfg):
        with open(self.path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    def update_metadata(self, **fields):
        self.metadata.update(fields)
        (self.path / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def start_history(self):
        with open(self.path / "history.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HISTORY_FIELDS)

    def append_history(self, row):
        """Строка дописывается сразу после эпохи: при сбое история остаётся на диске."""
        with open(self.path / "history.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([repr(row[k]) if isinstance(row[k], float) else row[k]
                                    for k in HISTORY_FIELDS])

    def write_predictions(self, parts):
        """parts: {имя части: (index, y_true, y_pred)}."""
        with open(self.path / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(PRED_FIELDS)
            for name, (index, y_true, y_pred) in parts.items():
                for i, t, p in zip(index, y_true, y_pred):
                    w.writerow([name, int(i), int(t), int(p)])


def compute_metrics(run_path):
    """Итоговые метрики считаются только из файлов запуска, а не из памяти."""
    run_path = Path(run_path)
    with open(run_path / "predictions.csv", encoding="utf-8") as f:
        preds = list(csv.DictReader(f))
    with open(run_path / "history.csv", encoding="utf-8") as f:
        history = list(csv.DictReader(f))

    metrics = {}
    for part in ("val", "test"):
        rows = [r for r in preds if r["split"] == part]
        if rows:
            correct = sum(r["y_true"] == r["y_pred"] for r in rows)
            metrics[f"{part}_acc"] = correct / len(rows)
            metrics[f"n_{part}"] = len(rows)
    last = history[-1]
    metrics["val_loss"] = float(last["val_loss"])
    metrics["epochs"] = len(history)
    metrics["train_time_s"] = sum(float(r["epoch_time"]) for r in history)
    return metrics


# ---------- MLflow ----------

def setup_mlflow(experiment):
    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    return mlflow


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
