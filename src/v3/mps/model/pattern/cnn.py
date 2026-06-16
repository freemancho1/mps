""" 
CNN1DPatternModel ─ Phase-2 학습 기반 시계열 패턴 모델 (결정3·결정4)

[역할]
  - Phase-1 RuleBasedPatternEngine(캔들 룰)을 보완·대체하는 학습 기반 패턴 모델
  - 0~1 상대 정규화된 OHLCV 시계열(PatternInput.ohlcv_series)을 1D-CNN에 입력해
    TripleBarrier 라벨(BUY·HOLD)을 예측
  - PatternModelPort 계약 (run(PatternInput, bars) → PatternSignal)을 만족하므로
    RuleBasedPatternEngine과 교체 가능
    
[결정3·결정4 내용]
  - 시계열 수치 입력은 가볍고 빠르며 봉의 정확한 값을 직접 학습
  - 룰 기반과 동일한 TripleBarrier 라벨을 공유해 수치 트랙과 직접 비교 가능
  - 비전(차트 이미지) 모델을 phase-3으로 분리 (PatternInput.chart_image 예약)
  
[출력]
  - source="CNN" (PatternSignal.source로 신호 출처 추적)
  - pattern_name="CNN_SEQ" (학습 모델은 명시적 패턴명이 없어 트랙 식별자로 사용)
"""
from __future__ import annotations 

import time 
import numpy as np 
import torch 
from pathlib import Path 
from typing import Optional 

from mps.config import cfg
from mps.core.types import Bar, PatternInput, PatternSignal 
from mps.core.types import SignalDirection, PatternName


class CNN1DNet(torch.nn.Module):
    """ 
    1D-CNN 시계열 분류기
      - 입력 [B, N, C] → 전치 → [B, C, N]
      - B: BatchSize, N: Timestamp 갯 수, C: 0~1로 정규화된 OHLCV값(5개)
    """
    def __init__(
        self,
        in_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        
        in_channels = in_channels or cfg.train.cnn_settings.in_channels
        num_classes = num_classes or cfg.train.cnn_settings.num_classes
        dropout = cfg.train.cnn_settings.dropout if dropout is None else dropout 
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1),      # [B, 64, 1]
        )
        
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),                 # [B, 64]
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)       # [B, N, C] → [B, C, N]
        return self.head(self.conv(x))
    
    
class CNN1DPatternModel:
    """ PatternModelPort 구현체 (Phase-2) """
    
    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        model_arch: Optional[dict] = None,
    ) -> None:
        self._device = torch.device(device or cfg.model.torch_device)
        self._model_arch = model_arch or cfg.train.cnn_settings.to_dict()
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
    
    def predict(self, inp: PatternInput) -> tuple[SignalDirection, float]:
        x = torch.from_numpy(np.ascontiguousarray(inp.ohlcv_series)).float()
        x = x.unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[0]
            cls = int(torch.argmax(probs).item())
            conf = float(probs[cls].item())
        direction: SignalDirection = cfg.data.idx2dir[cls]
        if direction == cfg.str.hold:
            conf = 0.0
        return direction, round(conf, 4)
    
    def run(self, inp: PatternInput, bars: list[Bar]) -> PatternSignal: 
        start_time = time.perf_counter()
        direction, confidence = self.predict(inp)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return PatternSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            dir=direction,
            confidence=confidence,
            pattern_name=cfg.str.cnn_seq,
            source=cfg.str.cnn,
            latency_ms=latency_ms
        )
        