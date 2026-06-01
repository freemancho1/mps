""" 
LSTMModel ─ Phase-2 학습 기반 수치 모델 (결정5: LSTM → Transformer → 앙상블)

[역할]
  - Phase-1 ThresholdModel(룰 기반)을 대체하는 학습 기반 수치 모델.
  - 120~240분 롤링 Z-score 피처(NumericInput.window)를 입력받아 
    TripleBarrier 라벨(BUY·SELL·HOLD)을 예측
  - NumericModelPort 계약 ─ run(NumericInput) → NumericSignal ─을 만족하므로,
    시뮬레이터·집행기 코드 변경 없이 ThresholdModel과 교체 가능
    
[출력]
  - direction: 소프트맥스 argmax 클래스 (BUY·SELL·HOLD)
  - confidence: 해당 클래스의 소프트맥스 확률 (HOLD는 0.0으로 강제)
  - feature_contrib: 'gradient X input' 기반 피처 기여도 (관측 가능성 원칙)
  
[재현 가능성]
  - 가중치는 파일에서 로드하며, 메타데이터(시드·아키텍처)와 함께 저장됨.
  - 가중치가 없으면 학습되지 않은 상태로 동작하므로, 
    팩토리에서 가중치 부재 시 Phase-1으로 풀백함.
"""
from __future__ import annotations 

import time 
import numpy as np 
import torch 
import torch.nn as nn

from pathlib import Path 
from typing import Optional

from mps.config import cfg

class LSTMNet(nn.Module):
    """ 
    시계열 분류 LSTM.
    
    입력 [B, N, F=14] (Z-score 정규화 피처)
      → LSTM (마지막 타임스텝 hidden state 사용)
      → FC head → [B, 3(=BUY·SELL·HOLD)]
    """
    def __init__(
        self, 
        input_size: int = cfg.lstm.input_size,
        hidden_size: int = cfg.lstm.hidden_size,
        num_layers: int = cfg.lstm.num_layers,
        num_classes: int = cfg.lstm.num_classes,
        dropout: float = cfg.lstm.dropout,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)   # out: [B, N, hidden]
        last = out[:, -1, :]    # 마지막 타임스탭 [B, hidden]
        return self.head(last)  # [B, num_classes]
    
    
class LSTMModel:
    """ NumericModelPort 구현체 (Phase-2) """
    def __init__(
        self,
        weights_path: Optional[Path] = None, 
        device: str = cfg.run.torch_device,
        
    )