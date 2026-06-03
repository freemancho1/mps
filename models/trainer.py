""" 
ModelTrainer ─ 학습 기반 모델(LSTM·1D-CNN) 공통 학습 루프

[설계]
  - 두 트랙 모두 3-클래스(BUY·SELL·HOLD) 분류이므로 학습 루프를 공유.
  - 시간 순 train·val 분할(누출 방지, 결정 8 정신) ─ 무작위 셔플 금지.
  - 클래스 불균형(HOLD 다수) 보정을 위해 CrosEntropyLoss에 클래스 가중치 적용.
  - 검증 손실 기준 조기 종료(early stopping)로 과적합 방지
  
[재현 가능성]
  - seed 고정(torch·numpy)으로 동일 데이터·코드·시드 → 동일 결과
"""
from __future__ import annotations 

import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Optional
from collections.abc import Sized

from mps.config import cfg, msg 
from mps.core.libs import set_seed
from mps.core.types import TrainHistory


def _time_split(dataset: Dataset, val_ratio: float) -> tuple[Subset, Subset]:
    """ 시간순 분할: 앞쪽(학습), 뒤쪽(검증) ⇒ 셔플 없음(누출 방지) """
    if not isinstance(dataset, Sized):
        raise TypeError(msg.training.err.not_len_func)
    
    num = len(dataset)
    num_val = max(1, int(num * val_ratio))
    num_train = num - num_val

    train_idx = list(range(0, num_train))
    val_idx = list(range(num_train, num))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def _class_weights(dataset: Dataset, device: torch.device) -> torch.Tensor:
    """ 클래스 빈도 역수 기반 가중치 (불균형 보정) """
    if not isinstance(dataset, Sized):
        raise TypeError(msg.training.err.not_len_func)
    
    # dataset에 "class_counts()" 속성·함수가 있으면 반환하고, 
    # 없으면 아무것도 안하는 함수를 반환
    counts = getattr(dataset, "class_counts", lambda: None)()
    if counts is None:
        labels = np.array([int(dataset[idx][1]) for idx in range(len(dataset))])
        counts = np.bincount(labels, minlength=3)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class ModelTrainer:
    def __init__(self) -> None:
        self._device = torch.device(cfg.train.device)

    def train(
        self, model: torch.nn.Module, dataset: Dataset
    ) -> tuple[torch.nn.Module, TrainHistory]:
        if not isinstance(dataset, Sized):
            raise TypeError(msg.training.err.not_len_func)
        
        if len(dataset) < 10:
            raise ValueError(msg.training.err.dataset_size(dataset))
        
        set_seed(cfg.train.seed)
        model = model.to(self._device)

        train_ds, val_ds = _time_split(dataset, cfg.train.val_ratio)
        # 학습셋은 셔플 가능(시간순 분할 후이므로 누출 없음), 검증셋은 순서 유지
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

        weights = _class_weights(train_ds, self._device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )

        history = TrainHistory()
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        # print(best_state)
        epochs_no_improve = 0

        for epoch in range(cfg.train.epochs):
            # ── 학습 ─────────────────
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self._device), batch_y.to(self._device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_X)
            train_loss /= len(train_ds)

            # ── 검증 ─────────────────
            model.eval()
            val_loss, correct = 0.0, 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self._device), batch_y.to(self._device)
                    logits = model(batch_X)
                    val_loss += criterion(logits, batch_y).item() * len(batch_X)
                    correct += (logits.argmax(dim=-1) == batch_y).sum().item()
            val_loss /= len(val_ds)
            val_acc = correct / len(val_ds)

            history.train_loss.append(round(train_loss, 5))
            history.val_loss.append(round(val_loss, 5))
            history.val_acc.append(round(val_acc, 4))

            print(msg.training.result_epoch(epoch, train_loss, val_loss, val_acc))

            # ── 조기 종료 판정 ────────────
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss 
                history.best_epoch = epoch 
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.train.patience:
                    break 

        model.load_state_dict(best_state)   # 최적 가중치 복원
        model.eval()
        return model, history
