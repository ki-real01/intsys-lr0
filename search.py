"""Эксперимент 2: поиск гиперпараметров в Optuna.

    python search.py --target 10     # первая часть поиска (обрыв после 10 trial)
    python search.py --target 40     # продолжение той же study до 40 trial
    python search.py --export-best   # сохранить лучший trial в configs/best.yaml

Study хранится в optuna.db, каждый trial — вложенная запись MLflow внутри
одной родительской записи "lr0_search".
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import optuna  # noqa: E402
import yaml  # noqa: E402
from optuna.study import MaxTrialsCallback  # noqa: E402

from lr0.config import deep_merge, load_config, validate  # noqa: E402
from lr0.data import load_data  # noqa: E402
from lr0.tracking import git_commit, require_clean_tree, setup_mlflow  # noqa: E402
from lr0.training import Trainer  # noqa: E402

MODEL_KEYS = ("n_layers", "hidden", "dropout")
TRAIN_KEYS = ("lr", "weight_decay", "batch_size")


def load_search_config(debug):
    with open("configs/search.yaml", encoding="utf-8") as f:
        sc = yaml.safe_load(f)
    if debug:
        sc["study_name"] += "_debug"
        sc["storage"] = "sqlite:///optuna_debug.db"
    return sc


def suggest(trial, space):
    params = {}
    for name, s in space.items():
        if s["type"] == "float":
            params[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        elif s["type"] == "int":
            params[name] = trial.suggest_int(name, s["low"], s["high"], log=s.get("log", False))
        elif s["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, s["choices"])
        else:
            raise ValueError(f"Неизвестный тип параметра {name}: {s['type']}")
    return params


def params_to_config(base_cfg, params, series, seed):
    override = {
        "series": series,
        "seed": seed,
        "model": {k: params[k] for k in MODEL_KEYS},
        "train": {k: params[k] for k in TRAIN_KEYS},
    }
    return validate(deep_merge(base_cfg, override))


def make_study(sc):
    return optuna.create_study(
        study_name=sc["study_name"], storage=sc["storage"], load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sc["sampler"]["seed"]),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=sc["pruner"]["n_startup_trials"],
            n_warmup_steps=sc["pruner"]["n_warmup_steps"],
        ),
    )


def find_parent_run(mlflow, study_name):
    """Родительская запись MLflow этой study, если поиск уже начинался."""
    runs = mlflow.search_runs(
        filter_string=f"tags.role = 'search_parent' and tags.study = '{study_name}'",
        output_format="list",
    )
    return runs[0].info.run_id if runs else None


def search(target, debug):
    sc = load_search_config(debug)
    commit = git_commit() if debug else require_clean_tree()
    base_cfg = load_config(sc["base_config"])
    data = load_data()
    study = make_study(sc)
    mlflow = setup_mlflow("lr0_debug" if debug else "lr0")

    def objective(trial):
        params = suggest(trial, sc["space"])
        cfg = params_to_config(base_cfg, params, "search", sc["seed"])
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.set_tags({"series": "search", "commit": commit, "trial_number": trial.number,
                             "study": sc["study_name"], "seed": sc["seed"]})
            mlflow.log_params(params)
            trainer = Trainer(cfg, data)
            epochs_done, t0, val_acc = 0, time.perf_counter(), float("nan")
            for row in trainer.epochs():
                epochs_done = row["epoch"]
                val_acc = row["val_acc"]
                trial.set_user_attr("epochs_done", epochs_done)
                trial.set_user_attr("train_time_s", time.perf_counter() - t0)
                mlflow.log_metrics({k: row[k] for k in ("train_loss", "val_loss", "val_acc")},
                                   step=row["epoch"])
                if math.isnan(row["train_loss"]):
                    # Обучение разошлось: trial получит состояние FAIL, поиск продолжится.
                    mlflow.set_tag("state", "FAIL")
                    return float("nan")
                trial.report(val_acc, step=row["epoch"])
                if trial.should_prune():
                    mlflow.set_tag("state", "PRUNED")
                    raise optuna.TrialPruned()
            mlflow.set_tag("state", "COMPLETE")
            mlflow.log_metric("final_val_acc", val_acc)
            return val_acc

    parent_id = find_parent_run(mlflow, sc["study_name"])
    with mlflow.start_run(run_id=parent_id, run_name=sc["study_name"]):
        mlflow.set_tags({"role": "search_parent", "study": sc["study_name"], "series": "search",
                         "commit": commit})
        before = len(study.trials)
        study.optimize(objective, n_trials=target,
                       callbacks=[MaxTrialsCallback(target, states=None)])
        mlflow.log_metric("n_trials", len(study.trials))
    print(f"Trial в study: было {before}, стало {len(study.trials)}")
    counts = {}
    for t in study.trials:
        counts[t.state.name] = counts.get(t.state.name, 0) + 1
    print("Состояния:", counts)
    print(f"Лучший trial #{study.best_trial.number}: val_acc={study.best_value:.4f} {study.best_params}")


def export_best(debug, path="configs/best.yaml"):
    sc = load_search_config(debug)
    study = make_study(sc)
    best = study.best_trial
    base_cfg = load_config(sc["base_config"])
    cfg = params_to_config(base_cfg, best.params, "best", 0)
    header = (f"# Лучший trial study {sc['study_name']}: #{best.number}, "
              f"val_acc в поиске = {best.value:.4f} (seed 0)\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"Сохранено в {path}:\n{header}{yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="ЛР0: поиск гиперпараметров в Optuna")
    p.add_argument("--target", type=int, help="довести study до стольких trial (по умолчанию 40)")
    p.add_argument("--export-best", action="store_true", help="записать лучший trial в configs/best.yaml")
    p.add_argument("--debug", action="store_true", help="отладка: optuna_debug.db, можно с грязным деревом")
    args = p.parse_args(argv)
    if args.export_best:
        export_best(args.debug)
    else:
        target = args.target or load_search_config(args.debug)["n_trials"]
        search(target, args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
