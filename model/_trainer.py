""" 
ModelTrainer ─ 학습 기반 모델(LSTM·1D-CNN) 공통 학습 트레이너

[설계]
- 두 트랙 모두 2-클래스(BUY·HOLD, 롱 온리) 분류로 학습 루프 공유 가능
- 시간순 train/val 분할 (셔플 금지) + 엠바고(purge) 적용
- 클래스 불균형(HOLD 다수) 보정: CrossEntropyLoss 클래스 가중치 사용
"""
from __future__ import annotations 

import torch 
import numpy as np 
from torch.utils.data import DataLoader, Dataset, Subset
from collections.abc import Sized 
from typing import Optional, TypeVar, cast 

from mps.core.config import cfg, msg 
from mps.core.libs import set_seed
from mps.core.libs import logger
from mps.core.types import TrainHistory

# Pylance 오류 무시용 
_NetModule = TypeVar("_NetType", bound=torch.nn.Module)


class ModelTrainer:
    def __init__(self, device: Optional[str] = None) -> None:
        self._device = torch.device(device or cfg.modeling.torch_device)

    def train(
        self, 
        model: _NetModule, 
        ds: Dataset, 
        epochs: Optional[int] = None,
    ) -> tuple[_NetModule, TrainHistory]:
        if not isinstance(ds, Sized):
            err_msg = msg.trainer.invalid_data_type(ds)
            logger.debug(msg.logger_like(err_msg))
            raise TypeError(err_msg)
        if len(ds) < cfg.modeling.min_dataset_size:
            err_msg = msg.trainer.insufficiend_data(
                ds, cfg.modeling.min_dataset_size
            )
            logger.debug(msg.logger_like(err_msg))
            raise ValueError(err_msg)
        self._epochs = cfg.params.epochs if epochs is None else epochs 
        
        set_seed(cfg.sys.seed)
        model = cast(_NetModule, model.to(self._device))

        train_ds, val_ds = _time_split(
            ds, cfg.params.val_ratio, cfg.data.embargo_bars
        )
        # train_ds는 시간순 분할 후에는 셔플하는것이 성능에 더 좋음
        train_loader = DataLoader(
            train_ds, batch_size=cfg.params.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.params.batch_size, shuffle=False
        )

        weights = _class_weights(train_ds, self._device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.params.lr, 
            weight_decay=cfg.params.weight_decay
        )

        history = TrainHistory()
        best_state = {
            key: value.detach().clone() 
            for key, value in model.state_dict().items()
        }
        epochs_no_improve = 0

        for epoch in range(self._epochs):
        # TODO 9999:9999 잠온다!!!


def _time_split(
    ds: Dataset, val_ratio: float, embargo: int
) -> tuple[Subset, Subset]:
    """ 
    시간순 분할 + 엠바고: [train ...][embargo 폐기][... val]

    - 엠바고가 train을 지나치게 깍으면(절반 이상) 데이터가 너무 작은 것이므로
      엠바고를 줄여서라도 최소 학습을 위한 데이터를 확보할 필요가 있음.
    - 이 함수는 이 파일 내부 함수로 ModelTrainer의 train()에서만 호출함.
    """
    if not isinstance(ds, Sized):
        logger.debug(msg.logger_like(msg.trainer.invalid_data_type(ds)))
        raise TypeError(msg.trainer.invalid_data_type(ds))
    
    ds_size = len(ds)
    val_size = max(1, int(ds_size * val_ratio))
    train_last_idx = ds_size - val_size 

    # 과도한 학습 데이터 손실을 방지하기 위해 최대 훈련 데이터의 1/2만 엠바고 지정
    effective_embargo = embargo 
    limit_embargo = train_last_idx // 2
    if embargo > limit_embargo:
        logger.info(msg.trainer.too_much_embargo(embargo, limit_embargo))
        effective_embargo = limit_embargo
    train_size = train_last_idx - effective_embargo

    train_idx = list(range(0, train_size))
    val_idx = list(range(train_last_idx, ds_size))
    return Subset(ds, train_idx), Subset(ds, val_idx)


def _class_weights(ds: Dataset, device: torch.device) -> torch.Tensor:
    """ 클래스 빈도 가중치 조정 (BUY 희소성을 4% 이상으로 보정) """
    if not isinstance(ds, Sized):
        logger.debug(msg.logger_like(msg.trainer.invalid_data_type(ds)))
        raise TypeError(msg.trainer.invalid_data_type(ds))

    counts = getattr(ds, cfg.key.class_counts, lambda: None)()
    if counts is None:
        labels = np.array([int(ds[idx][1]) for idx in range(len(ds))])    
        counts = np.bincount(labels, minlength=cfg.lstm.num_classes)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(counts) * counts)
    logger.debug(msg.trainer.class_calibration(weights))

    return torch.tensor(weights, dtype=torch.float32, device=device)
    



        
        
