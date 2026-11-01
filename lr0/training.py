"""Обучение и оценка. Одна и та же функция используется в train.py и в Optuna."""
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import MLP


def accuracy(y_true, y_pred):
    """Доля совпадений. Знаменатель — все записи, пустой вход — ошибка."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(f"accuracy: длины {len(y_true)} и {len(y_pred)} недопустимы")
    return float((y_true == y_pred).sum() / len(y_true))


def set_determinism(seed, num_threads):
    """Всё, что нужно, чтобы запуск с тем же seed повторялся на CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


class Trainer:
    def __init__(self, cfg, data):
        self.cfg = cfg
        m, t = cfg["model"], cfg["train"]
        set_determinism(cfg["seed"], cfg["runtime"]["num_threads"])

        X, y, split = data["X"], data["y"], data["split"]
        self.X = {p: torch.from_numpy(X[split[p]]) for p in split}
        self.y = {p: torch.from_numpy(y[split[p]]) for p in split}
        self.index = {p: split[p] for p in split}

        # Генератор с тем же seed задаёт порядок мини-батчей.
        gen = torch.Generator().manual_seed(cfg["seed"])
        self.loader = DataLoader(
            TensorDataset(self.X["train"], self.y["train"]),
            batch_size=t["batch_size"], shuffle=True, generator=gen, num_workers=0,
        )
        self.model = MLP(m["n_layers"], m["hidden"], m["dropout"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"]
        )
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _evaluate(self, part):
        self.model.eval()
        logits = self.model(self.X[part])
        loss = self.loss_fn(logits, self.y[part]).item()
        pred = logits.argmax(dim=1).numpy()
        return loss, pred

    def epochs(self):
        """Генератор: после каждой эпохи отдаёт строку истории.

        epoch_time — проход по train + оценка на val этой эпохи
        (запись файлов в интервал не входит, она снаружи генератора).
        """
        for epoch in range(1, self.cfg["train"]["epochs"] + 1):
            t0 = time.perf_counter()
            self.model.train()
            total, n = 0.0, 0
            for xb, yb in self.loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                total += loss.item() * len(yb)
                n += len(yb)
            val_loss, val_pred = self._evaluate("val")
            val_acc = accuracy(self.y["val"].numpy(), val_pred)
            epoch_time = time.perf_counter() - t0
            yield {
                "epoch": epoch,
                "train_loss": total / n,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time": epoch_time,
            }

    def predict(self, part):
        """Предсказания по части: (индекс в наборе, верный класс, ответ сети)."""
        _, pred = self._evaluate(part)
        return self.index[part], self.y[part].numpy(), pred
