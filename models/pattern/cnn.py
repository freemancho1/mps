""" 
CNN1DPatternModel ─ Phase-2 학습 기반 시계열 패턴 모델 (결정3·결정4)

[역할]
  - Phase-1 RuleBasedPatternEngine(캔들 룰)을 보완·대체하는 학습 기반 패턴 모델
  - 0~1 상대 정규화된 OHLCV 시계열(PatternInput.ohlcv_series)을 1D-CNN에 입력해
    Triple Barrier 라벨 (BUY·SELL·HOLD)을 예측
  - PatternModelPort 계약 (run(PatternInput, bars) → PatternSignal)을 만족하므로
    RuleBasedPatternEngine과 교체 가능

[결정3·4 근거]
  - 시계열 수치 입력은 가볍고 빠르며 봉의 정확한 값을 직접 학습.
  - 룰 기반과 동일한 Triple Barrier 라벨을 공유해 수치 트랙과 직접 비교 가능.
  - 비전(차트 이미지) 모델을 phase-3으로 분리 (PatternInput.chart_image 예약)

[출력]
  - source="CNN" (PatternSignal.source로 신호 출처 추적)
  - pattern_name="cnn_seq" (학습 모델은 명시적 패턴명이 없어 트랙 식별자로 사용)
"""
from __future__ import annotations

import time 
import numpy as np 
import torch 
from pathlib import Path 
from typing import Optional 

from mps.core.types import Bar, PatternInput, PatternSignal, Direction
from mps.pp.features.labeler import IDX_TO_LABEL
from mps.config import cfg, msg


class CNN1DNet(torch.nn.Module):
    """ 
    1D-CNN 시계열 분류기

    - 입력 [B, N, C=5] (0~1 정규화 OHLCV) → 전치 → [B, C, N]
    """
    def __init__(
        self,
        in_channels: int = cfg.cnn.in_channels,
        num_classes: int = cfg.cnn.num_classes,
        dropout: float = cfg.cnn.dropout,
    ) -> None:
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1),  # [B, 64, 1]
        )

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),             # [B, 64]
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)               # [B, N, C] → [B, C, N]
        return self.head(self.conv(x))
    

class CNN1DPatternModel:
    """ PatternModelPort 구현체 (Phase-2) """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: str = cfg.run.torch_device,     # cuda
        model_arch: Optional[dict] = None,
    ) -> None:
        self._device = torch.device(device)
        self._model_arch = model_arch or cfg.cnn.to_dict()
        self._model = CNN1DNet(**self._model_arch).to(self._device)
        self._trained = False

        if weights_path is not None and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self._device)
            state = ckpt[cfg.key.state_dict] if cfg.key.state_dict in ckpt else ckpt 
            self._model.load_state_dict(state)
            self._trained = True
        self._model.eval()

    @property 
    def model(self) -> CNN1DNet:
        return self._model 
    
    @property 
    def is_trained(self) -> bool:
        return self._trained 
    
    def predict(self, inp: PatternInput) -> tuple[Direction, float]:
        x = torch.from_numpy(np.ascontiguousarray(inp.ohlcv_series)).float()
        x = x.unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[0]
            cls = int(torch.argmax(probs).item())
            conf = float(probs[cls].item())
        direction: Direction = IDX_TO_LABEL[cls]
        if direction == cfg.key.HOLD:
            conf = 0.0
        return direction, round(conf, 4)
    
    def run(self, inp: PatternInput, bars: list[Bar]) -> PatternSignal:
        """ 
        추론 시간을 측정해 PatternSignal 생성.
        
        - bars 인자는 RuleBasedPatternEngine과의 인터페이스 호환을 위핸 받지만,
          CNN은 정규화된 ohlcv_series만 사용
        """
        start_time = time.perf_counter()
        direction, confidence = self.predict(inp)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return PatternSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            direction=direction,
            confidence=confidence,
            pattern_name="cnn_seq",
            source="CNN",
            latency_ms=latency_ms
        )