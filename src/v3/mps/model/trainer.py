""" 
ModelTrainer ─ 학습 기반 모델(LSTM·1D-CNN) 공통 학습 트레이너

[설계]
  - 두 트랙 모두 2-클래스(BUY·HOLD, 롱 온리) 분류로 학습 루프 공유 가능
  - 시간순 train/val 분할 (셔플 금지) + 엠바고(purge) 적용
  - 클래스 불균형(HOLD 다수) 보정: CrossEntropyLoss 클래스 가중치 사용
  - 검증 손실 기준 조기 종료
"""
from __future__ import annotations 

import numpy as np 
import torch 
from torch.utils.data import DataLoader, Dataset, Subset 
from collections.abc import Sized
from typing import Optional, TypeVar, cast

from mps.config import cfg, msg 
from mps.core.libs import set_seed
from mps.core.types import TrainHistory 
from mps.freelibs import logger 

# Pylance 경보 처리용
_NetT = TypeVar("_NetT", bound=torch.nn.Module)


class ModelTrainer:
    def __init__(self, device: Optional[str] = None) -> None:
        self._device = torch.device(cfg.model.torch_device if device is None else device)

    def train(self, model: _NetT, ds: Dataset) -> tuple[_NetT, TrainHistory]:
        if not isinstance(ds, Sized):
            raise TypeError(msg.training.err.not_len_func(ds))
        if len(ds) < cfg.model.min_dataset_size:
            raise ValueError(msg.training.err.insufficient_data(ds))
        
        set_seed(cfg.sys.seed)
        model = cast(_NetT, model.to(self._device))
        

        train_ds, val_ds = _time_split(
            ds, cfg.train.hyper_params.val_ratio, cfg.data.embargo_bars)
        # train은 시간순 분할 후이므로 셔플 가능, val은 셔플하면 안됨
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.hyper_params.batch_size, shuffle=True)
        val_loader = DataLoader(
            val_ds, batch_size=cfg.train.hyper_params.batch_size, shuffle=False)
        
        weights = _class_weights(train_ds, self._device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.hyper_params.lr, 
            weight_decay=cfg.train.hyper_params.weight_decay
        )

        history = TrainHistory()
        best_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        epochs_no_improve = 0

        for epoch in range(cfg.train.hyper_params.epochs):
            # ── 학습 ─────────────────────────
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
            
            # ── 검증 ─────────────────────────
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
            
            # logger.debug(msg.training.epoch_result(epoch, history))
            
            # ── 조기 종료 판정 ─────────────────────
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss 
                history.best_epoch = epoch 
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.train.hyper_params.patience:
                    break
                
        model.load_state_dict(best_state)
        model.eval()
        
        logger.debug(msg.training.result(model, history))
        return model, history
    

def _time_split(ds: Dataset, val_ratio: float, embargo: int) -> tuple[Subset, Subset]:
    """ 
    시간순 분할 + 엠바고: [train ...][embargo 폐기][... val]

    embargo가 train을 지나치게 깍으면(절반 이상) 데이터가 너무 작은 것이므로
    엠바고를 줄여서라도 최소 학습을 위한 데이터를 확보할 필요가 있음.
    """
    if not isinstance(ds, Sized):
        raise TypeError(msg.training.err.not_len_func(ds))
    
    ds_size = len(ds)
    val_size = max(1, int(ds_size * val_ratio))
    train_raw_size = ds_size - val_size

    effective_embargo = embargo
    limit_embargo = train_raw_size // 2
    if embargo > limit_embargo:     # 과도한 학습 데이터 손실 방지(최대 훈련 데이터의 1/2 이하만)
        logger.warning(msg.training.too_much_embargo(embargo, limit_embargo))
        effective_embargo = limit_embargo
    train_size = train_raw_size - effective_embargo

    train_idx = list(range(0, train_size))
    val_idx = list(range(train_raw_size, ds_size))
    return Subset(ds, train_idx), Subset(ds, val_idx)


def _class_weights(ds: Dataset, device: torch.device) -> torch.Tensor:
    """ 클래스 빈도 역수 기반 가중치 (BUY 희소성 ~4% 보정) """
    if not isinstance(ds, Sized):
        raise TypeError(msg.training.err.not_len_func(ds))
    
    counts = getattr(ds, cfg.key.class_counts, lambda: None)()
    if counts is None:
        labels = np.array([int(ds[idx][1]) for idx in range(len(ds))])
        counts = np.bincount(labels, minlength=cfg.train.lstm_settings.num_classes)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(counts) * counts)
    logger.debug(msg.training.class_calibration(weights))

    return torch.tensor(weights, dtype=torch.float32, device=device)