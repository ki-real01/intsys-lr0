# ЛР0. Воспроизводимый эксперимент: подбор гиперпараметров нейросети

Курс «Интеллектуальные системы», задание: [dm780009/intsys](https://github.com/dm780009/intsys), `LR0_experiment_harness.pdf`.

**Вопрос работы:** дал ли подбор гиперпараметров (Optuna) улучшение, которое больше случайного разброса от seed?

Объект — MLP на `sklearn.datasets.load_digits` (1797 изображений 8×8, 10 классов).
Прогноз — [hypothesis.md](hypothesis.md).

## Структура

| Файл | Что делает |
|------|------------|
| `prepare_data.py` | разбиение train/val/test = 1077/360/360 (`split_seed=0`) → `data/split.json`, `data/manifest.json` с SHA-256 |
| `lr0/config.py` | значения по умолчанию, слияние конфигурации с аргументами, проверка (например, `lr <= 0` отклоняется) |
| `lr0/data.py` | загрузка данных, сверка SHA-256 `split.json` с манифестом, проверка разбиения |
| `lr0/model.py` | MLP: 64 → n_layers × (Linear, ReLU, Dropout) → 10 |
| `lr0/training.py` | детерминизм (seed, генератор DataLoader, `use_deterministic_algorithms`, 1 поток) и цикл обучения по эпохам |
| `lr0/tracking.py` | git (коммит, проверка чистого дерева), каталог запуска, метрики из файлов, MLflow |
| `train.py` | **один запуск** по конфигурации |
| `run_series.py` | серия запусков, каждый — отдельный процесс `train.py` |
| `search.py` | поиск Optuna (TPE seed 42, MedianPruner, 40 trial) и экспорт лучшего trial |
| `configs/` | `base.yaml`, `search.yaml` (пространство поиска), `best.yaml` (после поиска) |
| `tests/` | проверки реализации |

Каталог запуска `runs/<run_id>/`: `config.yaml` (итоговая конфигурация), `metadata.json`
(статус, время, коммит, версии, seed, SHA-256 данных, id записи MLflow),
`history.csv` (по эпохам, дописывается после каждой эпохи), `predictions.csv`
(ответ на каждое изображение), `metrics.json` (пересчитано из `predictions.csv` и `history.csv`).

## Установка (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_data.py      # проверит, что data/split.json совпадает
python -m pytest -q         # проверки реализации
```

## Запуск экспериментов

Итоговые запуски работают только из закоммиченного кода (иначе отказ). Флаг `--debug` — отладка, пишет в `runs_debug/` и эксперимент MLflow `lr0_debug`.

```powershell
# Эксперимент 1
python run_series.py --series repeat_seed0 --config configs/base.yaml --seeds 0 0
python run_series.py --series base_seeds   --config configs/base.yaml --seeds 1 2 3 4 5

# Эксперимент 2: поиск с обрывом после 10 trial и продолжением до 40
python search.py --target 10
python search.py --target 40
python search.py --export-best          # -> configs/best.yaml, затем закоммитить

# Эксперимент 3: test используется только здесь
python run_series.py --series final --config configs/base.yaml configs/best.yaml --seeds 101 102 103 104 105 --eval-test
```

Один запуск вручную: `python train.py --config configs/base.yaml --seed 0 [--eval-test]`.

## MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Открыть http://127.0.0.1:5000, эксперимент `lr0`. Журнал — `mlflow.db`, файлы-артефакты — `mlruns/`; вместе с `runs/` и `optuna.db` они не попадают в git и сдаются архивом.
