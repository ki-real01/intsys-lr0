"""Проверки реализации (п. 3.4 ЛР0). Их результаты в таблицы отчёта не входят.

    python -m pytest -q

Запуски выполняются во временной копии проекта со своим git-репозиторием,
поэтому runs/ и mlflow.db основного проекта не затрагиваются.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lr0.config import ConfigError, load_config  # noqa: E402
from lr0.training import accuracy  # noqa: E402

COPY_ITEMS = ["lr0", "configs", "data", "train.py", "run_series.py", "search.py",
              "prepare_data.py", ".gitignore"]


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Чистая копия проекта, закоммиченная в отдельный git-репозиторий."""
    dst = tmp_path_factory.mktemp("proj")
    for item in COPY_ITEMS:
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, dst / item, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst / item)
    git(dst, "init", "-q")
    git(dst, "add", ".")
    git(dst, "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "init")
    return dst


def train(project, *args):
    # PYTHONUTF8: на Windows дочерний процесс иначе пишет stderr в cp1251.
    env = {**os.environ, "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, "train.py", *args], cwd=project, env=env,
                          capture_output=True, text=True, encoding="utf-8")


def last_run_dir(project, before):
    after = set((project / "runs").iterdir())
    new = after - before
    assert len(new) == 1, new
    return new.pop()


def run_and_get_dir(project, *args):
    runs = project / "runs"
    before = set(runs.iterdir()) if runs.exists() else set()
    res = train(project, *args)
    assert res.returncode == 0, res.stderr
    return last_run_dir(project, before)


def read_csv(path, drop=()):
    with open(path, encoding="utf-8") as f:
        return [{k: v for k, v in row.items() if k not in drop} for row in csv.DictReader(f)]


# ---------- проверки ----------

def test_accuracy_known_example():
    assert accuracy([1, 2, 3, 4], [1, 2, 3, 3]) == 0.75


def test_accuracy_empty_is_error():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_negative_lr_rejected_before_training(project, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("train: {lr: -1}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="train.lr"):
        load_config(bad)
    runs = project / "runs"
    before = set(runs.iterdir()) if runs.exists() else set()
    res = train(project, "--config", str(bad))
    assert res.returncode == 2
    assert "train.lr должен быть > 0" in res.stderr
    after = set(runs.iterdir()) if runs.exists() else set()
    assert after == before, "каталог запуска не должен создаваться"


def test_split_is_valid(project):
    split = json.loads((project / "data/split.json").read_text(encoding="utf-8"))
    sizes = [len(split[p]) for p in ("train", "val", "test")]
    assert sizes == [1077, 360, 360]
    all_ids = split["train"] + split["val"] + split["test"]
    assert len(set(all_ids)) == 1797


def test_same_seed_same_results_other_seed_differs(project):
    a = run_and_get_dir(project, "--config", "configs/base.yaml", "--seed", "0")
    b = run_and_get_dir(project, "--config", "configs/base.yaml", "--seed", "0")
    c = run_and_get_dir(project, "--config", "configs/base.yaml", "--seed", "1")
    # epoch_time — время, оно не обязано совпадать; всё остальное должно совпасть до бита.
    ha, hb, hc = (read_csv(d / "history.csv", drop=("epoch_time",)) for d in (a, b, c))
    pa, pb, pc = (read_csv(d / "predictions.csv") for d in (a, b, c))
    assert ha == hb
    assert pa == pb
    assert ha != hc


def test_metrics_recomputed_from_predictions(project):
    d = run_and_get_dir(project, "--config", "configs/base.yaml", "--seed", "2", "--eval-test")
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    preds = read_csv(d / "predictions.csv")
    history = read_csv(d / "history.csv")
    val = [r for r in preds if r["split"] == "val"]
    test = [r for r in preds if r["split"] == "test"]
    assert len(val) == 360 and len(test) == 360
    recomputed = sum(r["y_true"] == r["y_pred"] for r in val) / len(val)
    assert metrics["val_acc"] == pytest.approx(recomputed)
    # та же точность, что была посчитана в памяти во время обучения
    assert float(history[-1]["val_acc"]) == pytest.approx(metrics["val_acc"])
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert len(meta["commit"]) == 40


def test_no_test_predictions_without_flag(project):
    d = run_and_get_dir(project, "--config", "configs/base.yaml", "--seed", "3")
    assert {r["split"] for r in read_csv(d / "predictions.csv")} == {"val"}


def test_dirty_tree_refuses_final_run(project):
    f = project / "lr0" / "model.py"
    original = f.read_text(encoding="utf-8")
    try:
        f.write_text(original + "\n# изменение\n", encoding="utf-8")
        res = train(project, "--config", "configs/base.yaml")
        assert res.returncode != 0
        assert "незакоммиченные изменения" in res.stderr
        res_debug = train(project, "--config", "configs/base.yaml", "--debug")
        assert res_debug.returncode == 0, res_debug.stderr
    finally:
        f.write_text(original, encoding="utf-8")
        shutil.rmtree(project / "runs_debug", ignore_errors=True)
