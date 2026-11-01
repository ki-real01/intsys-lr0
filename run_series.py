"""Серия запусков: каждый запуск — отдельный процесс train.py.

    python run_series.py --series repeat_seed0 --config configs/base.yaml --seeds 0 0
    python run_series.py --series base_seeds   --config configs/base.yaml --seeds 1 2 3 4 5
    python run_series.py --series final --config configs/base.yaml configs/best.yaml \
        --seeds 101 102 103 104 105 --eval-test

Упавший запуск остаётся в runs/ со статусом failed, серия продолжается.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from lr0.tracking import require_clean_tree  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="ЛР0: серия запусков")
    p.add_argument("--series", required=True)
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--eval-test", action="store_true")
    args = p.parse_args(argv)

    require_clean_tree()
    plan = [(c, s) for c in args.config for s in args.seeds]
    print(f"Серия {args.series}: {len(plan)} запусков")
    failed = 0
    for i, (config, seed) in enumerate(plan, 1):
        cmd = [sys.executable, "train.py", "--config", config, "--seed", str(seed),
               "--series", args.series]
        if args.eval_test:
            cmd.append("--eval-test")
        print(f"[{i}/{len(plan)}] {' '.join(cmd[1:])}", flush=True)
        code = subprocess.run(cmd).returncode
        if code != 0:
            failed += 1
            print(f"  запуск завершился с кодом {code}")
    print(f"Готово: {len(plan) - failed} успешно, {failed} с ошибкой")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
