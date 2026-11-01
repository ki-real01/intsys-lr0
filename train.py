"""Один запуск: обучение и оценка по одной конфигурации.

    python train.py --config configs/base.yaml [--seed N] [--series NAME] [--eval-test] [--debug]

Итоговый запуск (без --debug) требует чистого git-дерева и пишет в runs/ и
эксперимент MLflow "lr0". Отладочный (--debug) — в runs_debug/ и "lr0_debug".
"""
import argparse
import json
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from lr0.config import ConfigError, load_config  # noqa: E402
from lr0.data import load_data  # noqa: E402
from lr0.tracking import (  # noqa: E402
    RunDir, compute_metrics, flatten, git_commit, git_dirty, library_versions,
    now_iso, require_clean_tree, setup_mlflow,
)
from lr0.training import Trainer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ЛР0: один запуск обучения MLP")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, help="переопределяет seed из конфигурации")
    p.add_argument("--series", help="имя серии (по умолчанию из конфигурации)")
    p.add_argument("--eval-test", action="store_true", help="оценить и на test")
    p.add_argument("--debug", action="store_true", help="отладочный запуск: можно с грязным деревом")
    return p.parse_args(argv)


def run(cfg, eval_test=False, debug=False):
    """Выполняет запуск и возвращает путь к его каталогу."""
    commit = git_commit()
    dirty = git_dirty()
    if not debug:
        require_clean_tree()

    runs_root = "runs_debug" if debug else "runs"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['series']}-s{cfg['seed']}-{uuid.uuid4().hex[:6]}"
    rd = RunDir(runs_root, run_id)
    rd.write_config(cfg)
    rd.update_metadata(
        run_id=run_id, experiment=cfg["experiment"], series=cfg["series"],
        status="running", debug=debug, started_at=now_iso(), finished_at=None,
        commit=commit, dirty_files=dirty.splitlines() if dirty else [],
        seed=cfg["seed"], split_seed=cfg["split_seed"], eval_test=eval_test,
        device=cfg["runtime"]["device"], num_threads=cfg["runtime"]["num_threads"],
        versions=library_versions(), command=" ".join(sys.argv),
    )

    mlflow = setup_mlflow("lr0_debug" if debug else "lr0")
    with mlflow.start_run(run_name=run_id) as mlrun:
        mlflow.set_tags({
            "experiment": cfg["experiment"], "series": cfg["series"], "commit": commit,
            "seed": cfg["seed"], "run_id": run_id, "eval_test": eval_test,
        })
        mlflow.log_params(flatten(cfg))
        rd.update_metadata(mlflow_run_id=mlrun.info.run_id)
        try:
            data = load_data()
            rd.update_metadata(split_sha256=data["split_sha256"])
            trainer = Trainer(cfg, data)

            rd.start_history()
            for row in trainer.epochs():
                rd.append_history(row)
                mlflow.log_metrics(
                    {k: row[k] for k in ("train_loss", "val_loss", "val_acc", "epoch_time")},
                    step=row["epoch"],
                )

            parts = {"val": trainer.predict("val")}
            if eval_test:
                parts["test"] = trainer.predict("test")
            rd.write_predictions(parts)

            metrics = compute_metrics(rd.path)
            (rd.path / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})
            rd.update_metadata(status="completed", finished_at=now_iso())
        except Exception as exc:
            rd.update_metadata(
                status="failed", finished_at=now_iso(),
                error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            )
            mlflow.set_tag("status", "failed")
            mlflow.log_artifacts(str(rd.path))
            raise
        mlflow.set_tag("status", "completed")
        mlflow.log_artifacts(str(rd.path))
    return rd.path


def main(argv=None):
    args = parse_args(argv)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.series:
        overrides["series"] = args.series
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    path = run(cfg, eval_test=args.eval_test, debug=args.debug)
    m = (path / "metrics.json").read_text(encoding="utf-8")
    print(f"Запуск завершён: {path}\n{m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
