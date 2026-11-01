"""Таблицы, графики и отчёт строятся только из сохранённых результатов.

    python make_report.py

Читает runs/*/ (metadata, history, predictions, metrics), optuna.db и mlflow.db.
Пишет report/*.csv, report/*.png, report/numbers.json и собирает report/REPORT.md
из шаблона report/REPORT_template.md (числа подставляются скриптом, а не вручную).
Рисунки и таблицы также сохраняются в MLflow (запись "report").
"""
import csv
import json
import os
import statistics
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from lr0.tracking import setup_mlflow  # noqa: E402

OUT = Path("report")
STUDY = "lr0_search"
STORAGE = "sqlite:///optuna.db"


# ---------- чтение запусков ----------

def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_runs():
    runs = []
    for d in sorted(Path("runs").iterdir()):
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        run = {"dir": d, "meta": meta}
        if meta["status"] == "completed":
            run["metrics"] = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        runs.append(run)
    return runs


def config_label(run):
    """base или best — по параметрам модели в итоговой конфигурации запуска."""
    text = (run["dir"] / "config.yaml").read_text(encoding="utf-8")
    return "base" if "hidden: 64" in text and "n_layers: 1" in text else "best"


def preds_by_index(run, split):
    return {int(r["index"]): (r["y_true"] == r["y_pred"])
            for r in read_csv(run["dir"] / "predictions.csv") if r["split"] == split}


def fixed_broken(a, b):
    """Сколько записей B исправил по сравнению с A и сколько испортил."""
    assert a.keys() == b.keys()
    fixed = sum(1 for i in a if not a[i] and b[i])
    broken = sum(1 for i in a if a[i] and not b[i])
    return fixed, broken


def pp(x):
    return f"{x * 100:.2f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------- эксперимент 1 ----------

def experiment1(runs, num):
    rep = [r for r in runs if r["meta"]["series"] == "repeat_seed0"]
    assert len(rep) == 2 and all(r["meta"]["status"] == "completed" for r in rep)
    hist = [[{k: v for k, v in row.items() if k != "epoch_time"}
             for row in read_csv(r["dir"] / "history.csv")] for r in rep]
    preds = [read_csv(r["dir"] / "predictions.csv") for r in rep]
    times = [[row["epoch_time"] for row in read_csv(r["dir"] / "history.csv")] for r in rep]
    num["repeat_history_equal"] = hist[0] == hist[1]
    num["repeat_predictions_equal"] = preds[0] == preds[1]
    num["repeat_epoch_time_equal"] = times[0] == times[1]
    num["repeat_val_acc"] = pp(rep[0]["metrics"]["val_acc"])
    num["repeat_run_ids"] = ", ".join(f"`{r['meta']['run_id']}`" for r in rep)

    series = sorted([r for r in runs if r["meta"]["series"] == "base_seeds"], key=lambda r: r["meta"]["seed"])
    planned = 5
    rows = []
    for r in series:
        m = r.get("metrics", {})
        rows.append({
            "seed": r["meta"]["seed"], "run_id": f"`{r['meta']['run_id']}`", "статус": r["meta"]["status"],
            "val_acc, %": pp(m["val_acc"]) if m else "—",
            "верно из 360": round(m["val_acc"] * m["n_val"]) if m else "—",
            "время обучения, с": f"{m['train_time_s']:.2f}" if m else "—",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp1_base_seeds.csv", index=False)
    accs = [r["metrics"]["val_acc"] for r in series if r["meta"]["status"] == "completed"]
    tts = [r["metrics"]["train_time_s"] for r in series if r["meta"]["status"] == "completed"]
    num.update({
        "exp1_table": md_table(df),
        "exp1_planned": planned, "exp1_completed": len(accs),
        "exp1_min": pp(min(accs)), "exp1_median": pp(statistics.median(accs)), "exp1_max": pp(max(accs)),
        "exp1_range_pp": f"{(max(accs) - min(accs)) * 100:.2f}",
        "exp1_range_imgs": round((max(accs) - min(accs)) * 360),
        "exp1_time_median": f"{statistics.median(tts):.2f}",
        "exp1_time_min": f"{min(tts):.2f}", "exp1_time_max": f"{max(tts):.2f}",
    })
    # Разные ошибки при близкой точности: seed 1 против seed 2.
    a, b = (preds_by_index(r, "val") for r in series[:2])
    fixed, broken = fixed_broken(a, b)
    num.update({"exp1_cmp_a": series[0]["meta"]["seed"], "exp1_cmp_b": series[1]["meta"]["seed"],
                "exp1_a_only": broken, "exp1_b_only": fixed,
                "exp1_both_wrong": sum(1 for i in a if not a[i] and not b[i])})


# ---------- эксперимент 2 ----------

def experiment2(num):
    study = optuna.load_study(study_name=STUDY, storage=STORAGE)
    trials = study.trials
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df.to_csv(OUT / "exp2_trials.csv", index=False)

    states = {}
    for t in trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    epochs_planned = 30 * len(trials)
    epochs_done = sum(t.user_attrs.get("epochs_done", 0) for t in trials)
    saved_s, spent_s = 0.0, 0.0
    for t in trials:
        done = t.user_attrs.get("epochs_done", 0)
        spent = t.user_attrs.get("train_time_s", 0.0)
        spent_s += spent
        if t.state == optuna.trial.TrialState.PRUNED and done:
            saved_s += spent / done * (30 - done)
    pruned_epochs = [t.user_attrs.get("epochs_done") for t in trials
                     if t.state == optuna.trial.TrialState.PRUNED]
    best = study.best_trial
    num.update({
        "exp2_n_trials": len(trials),
        "exp2_states": ", ".join(f"{k} — {v}" for k, v in sorted(states.items())),
        "exp2_complete": states.get("COMPLETE", 0), "exp2_pruned": states.get("PRUNED", 0),
        "exp2_fail": states.get("FAIL", 0),
        "exp2_epochs_done": epochs_done, "exp2_epochs_planned": epochs_planned,
        "exp2_epochs_saved_pct": f"{(1 - epochs_done / epochs_planned) * 100:.1f}",
        "exp2_time_spent": f"{spent_s:.1f}", "exp2_time_saved": f"{saved_s:.1f}",
        "exp2_time_saved_pct": f"{saved_s / (spent_s + saved_s) * 100:.1f}",
        "exp2_pruned_epochs": f"{min(pruned_epochs)}–{max(pruned_epochs)}" if pruned_epochs else "—",
        "exp2_best_number": best.number, "exp2_best_val": pp(best.value),
        "exp2_best_params": ", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}"
                                      for k, v in best.params.items()),
    })

    # Рисунок 1: точность каждого trial и лучшее значение на данный момент.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = {"COMPLETE": "#1f77b4", "PRUNED": "#ff7f0e", "FAIL": "#d62728"}
    for state, color in colors.items():
        xs = [t.number for t in trials if t.state.name == state]
        ys = [list(t.intermediate_values.values())[-1] * 100 if t.intermediate_values else float("nan")
              for t in trials if t.state.name == state]
        if xs:
            label = state if state != "PRUNED" else "PRUNED (точность на эпохе остановки)"
            ax.scatter(xs, ys, c=color, label=label, s=28, zorder=3)
    best_so_far, cur = [], float("-inf")
    for t in trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            cur = max(cur, t.value)
        best_so_far.append(cur * 100 if cur > float("-inf") else float("nan"))
    ax.step(range(len(trials)), best_so_far, where="post", c="black", lw=1.5, label="лучшее на данный момент")
    ax.axvline(9.5, ls="--", c="gray", lw=1)
    ax.text(9.7, ax.get_ylim()[0] + 0.5, "обрыв и продолжение", color="gray", fontsize=8)
    ax.set_xlabel("номер trial")
    ax.set_ylabel("точность на val, %")
    ax.set_title("Поиск Optuna lr0_search: точность trial на val")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_optuna_history.png", dpi=150)
    plt.close(fig)

    # Рисунок 2: важность параметров (fANOVA с фиксированным seed — повторяемо).
    imp = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    pd.DataFrame({"param": list(imp), "importance": list(imp.values())}).to_csv(
        OUT / "exp2_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    names = list(imp)[::-1]
    ax.barh(names, [imp[n] for n in names], color="#1f77b4")
    ax.set_xlabel("важность (fANOVA)")
    ax.set_title("Важность гиперпараметров")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "fig_param_importance.png", dpi=150)
    plt.close(fig)
    # Два худших остановленных trial — для разбора примеров.
    pruned = sorted((t for t in trials if t.state == optuna.trial.TrialState.PRUNED),
                    key=lambda t: list(t.intermediate_values.values())[-1])[:2]
    num["exp2_worst_pruned"] = "; ".join(
        f"trial {t.number}: {list(t.intermediate_values.values())[-1] * 100:.1f}% на эпохе "
        f"{t.user_attrs['epochs_done']}, lr = {t.params['lr']:.3g}, dropout = {t.params['dropout']:.2f}"
        for t in pruned)
    num["exp2_importance"] = ", ".join(f"{k} — {v:.2f}" for k, v in imp.items())
    num["exp2_top_param"] = list(imp)[0]


# ---------- эксперимент 3 ----------

def experiment3(runs, num):
    final = [r for r in runs if r["meta"]["series"] == "final"]
    assert all(r["meta"]["status"] == "completed" for r in final), "в серии final есть упавшие запуски"
    by = {(config_label(r), r["meta"]["seed"]): r for r in final}
    seeds = sorted({s for _, s in by})
    rows, dval, dtest = [], [], []
    fixed_total, broken_total = 0, 0
    for s in seeds:
        base, best = by[("base", s)], by[("best", s)]
        bm, km = base["metrics"], best["metrics"]
        f, b = fixed_broken(preds_by_index(base, "test"), preds_by_index(best, "test"))
        fixed_total += f
        broken_total += b
        dv, dt = km["val_acc"] - bm["val_acc"], km["test_acc"] - bm["test_acc"]
        dval.append(dv)
        dtest.append(dt)
        num[f"s{s}_fixed"], num[f"s{s}_broken"] = f, b
        num[f"s{s}_dtest"] = f"{dt * 100:+.2f}"
        num[f"s{s}_dtest_imgs"] = f - b
        rows.append({
            "seed": s,
            "base val, %": pp(bm["val_acc"]), "best val, %": pp(km["val_acc"]), "Δ val, п.п.": f"{dv * 100:+.2f}",
            "base test, %": pp(bm["test_acc"]), "best test, %": pp(km["test_acc"]), "Δ test, п.п.": f"{dt * 100:+.2f}",
            "test: исправлено / испорчено": f"{f} / {b}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp3_final.csv", index=False)
    best_val = [by[("best", s)]["metrics"]["val_acc"] for s in seeds]
    best_test = [by[("best", s)]["metrics"]["test_acc"] for s in seeds]
    base_test = [by[("base", s)]["metrics"]["test_acc"] for s in seeds]
    base_val = [by[("base", s)]["metrics"]["val_acc"] for s in seeds]
    study = optuna.load_study(study_name=STUDY, storage=STORAGE)
    num.update({
        "exp3_table": md_table(df), "exp3_n_runs": len(final),
        "exp3_seeds": f"{seeds[0]}–{seeds[-1]}",
        "exp3_dval_median": f"{statistics.median(dval) * 100:+.2f}",
        "exp3_dval_range": f"{min(dval) * 100:+.2f} … {max(dval) * 100:+.2f}",
        "exp3_dtest_median": f"{statistics.median(dtest) * 100:+.2f}",
        "exp3_dtest_range": f"{min(dtest) * 100:+.2f} … {max(dtest) * 100:+.2f}",
        "exp3_dtest_positive": sum(d > 0 for d in dtest),
        # разброс по seed из эксперимента 1 (max - min точности base_seeds на val)
        "exp3_dtest_above_spread": sum(d * 100 > float(num["exp1_range_pp"]) + 1e-9 for d in dtest),
        "exp3_dtest_mean": f"{statistics.mean(dtest) * 100:+.2f}",
        "exp3_dtest_median_imgs": round(statistics.median(dtest) * 360),
        "exp3_dtest_min_pp": f"{min(dtest) * 100:.2f}",
        "exp3_fixed": fixed_total, "exp3_broken": broken_total,
        "exp3_n_test_total": 360 * len(seeds),
        "exp3_search_best": pp(study.best_value),
        "exp3_best_val_mean": pp(statistics.mean(best_val)),
        "exp3_best_test_mean": pp(statistics.mean(best_test)),
        "exp3_base_val_mean": pp(statistics.mean(base_val)),
        "exp3_base_test_mean": pp(statistics.mean(base_test)),
        "exp3_gap_search_val": f"{(study.best_value - statistics.mean(best_val)) * 100:.2f}",
        "exp3_gap_val_test": f"{(statistics.mean(best_val) - statistics.mean(best_test)) * 100:.2f}",
    })

    # Рисунок 3: точность base и best на test по seed.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = range(len(seeds))
    ax.plot(x, [v * 100 for v in base_test], "o-", label="base")
    ax.plot(x, [v * 100 for v in best_test], "s-", label="best")
    ax.set_xticks(list(x), [str(s) for s in seeds])
    ax.set_xlabel("seed обучения")
    ax.set_ylabel("точность на test, %")
    ax.set_title("Серия final: base и best на одних и тех же seed")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_final_test.png", dpi=150)
    plt.close(fig)


def export_mlflow(num):
    mlflow = setup_mlflow("lr0")
    df = mlflow.search_runs(experiment_names=["lr0"])
    df.to_csv(OUT / "mlflow_runs.csv", index=False)
    num["mlflow_n_runs"] = len(df)
    return mlflow


def main():
    OUT.mkdir(exist_ok=True)
    runs = load_runs()
    num = {"n_runs_dirs": len(runs)}
    experiment1(runs, num)
    experiment2(num)
    experiment3(runs, num)
    mlflow = export_mlflow(num)

    text_nums = {k: v for k, v in num.items() if not k.endswith("_table")}
    (OUT / "numbers.json").write_text(json.dumps(text_nums, ensure_ascii=False, indent=2), encoding="utf-8")

    template = OUT / "REPORT_template.md"
    if template.exists():
        report = Template(template.read_text(encoding="utf-8")).substitute(num)
        (OUT / "REPORT.md").write_text(report, encoding="utf-8")
        print("Собран report/REPORT.md")

    # Рисунки и таблицы отчёта — тоже в MLflow, отдельной записью.
    for old in mlflow.search_runs(experiment_names=["lr0"], filter_string="tags.role = 'report'",
                                  output_format="list"):
        mlflow.delete_run(old.info.run_id)
    with mlflow.start_run(run_name="report"):
        mlflow.set_tag("role", "report")
        mlflow.log_artifacts(str(OUT), artifact_path="report")
    print(json.dumps(text_nums, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
