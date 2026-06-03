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

from mps.config import cfg, msg
from mps.core.types import Direction
from mps.core.types import NumericInput, NumericSignal
from mps.pp.features.labeler import IDX_TO_LABEL, LABEL_TO_IDX


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
        device: str = cfg.run.torch_device,     # cuda
        model_arch: Optional[dict] = None,
        attribute: bool = True,
    ) -> None:
        self._device = torch.device(device)
        self._model_arch = model_arch or cfg.lstm.to_dict()
        self._model = LSTMNet(**self._model_arch).to(self._device)
        self._attribute = attribute
        self._trained = False 

        if weights_path is not None and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self._device)
            state = ckpt[cfg.key.state_dict] if cfg.key.state_dict in ckpt else ckpt 
            self._model.load_state_dict(state)
            self._trained = True 
        
        self._model.eval()

    @property 
    def model(self) -> LSTMNet:
        return self._model
    
    @property 
    def is_trained(self) -> bool:
        return self._trained
    
    def predict(self, inp: NumericInput) -> tuple[Direction, float, dict]:
        x = torch.from_numpy(np.ascontiguousarray(inp.window)).float()
        x = x.unsqueeze(0).to(self._device)     # [1, N, 14]

        # 1차: no_grad 빠른 추론으로 방향·신뢰도 결정
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=-1)[0]
            cls = int(torch.argmax(probs).item())
            conf = float(probs[cls].item())

        direction: Direction = IDX_TO_LABEL[cls]
        if direction == cfg.key.HOLD:
            return cfg.run.no_signal    # "HOLD", 0.0, {} → 신호 미발생
        
        # 2차: 실제 신호(BUY·SELL)가 발생한 경우에만 어트리뷰션 계산
        #      → 매 봉이 아니라 신호 봉에서만 backward 1회 ⇒ 백테스트 속도 확보
        contrib: dict = {}
        if self._attribute:
            x.requires_grad_(True)
            logits = self._model(x)
            self._model.zero_grad(set_to_none=True)
            logits[0, cls].backward()
            # 아래 x.grad가 None인 경우 Pylance 오류 때문에 추가
            assert x.grad is not None, msg.training.not_compute_gradient
            saliency = (x.grad[0].abs() * x[0].abs()).sum(dim=0)
            contrib = LSTMModel._build_contrib(saliency.detach().cpu().numpy())
            
        return direction, round(conf, 4), contrib
    
    def run(self, inp: NumericInput) -> NumericSignal:
        """ 추론 시간을 측정해 NumericSignal 생성 """
        start_time = time.perf_counter()
        direction, confidence, contrib = self.predict(inp)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return NumericSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            direction=direction,
            confidence=confidence,
            feature_contrib=contrib,
            latency_ms=latency_ms
        )

    @staticmethod
    def _build_contrib(saliency: np.ndarray) -> dict:
        names = cfg.run.feature_names
        total = float(saliency.sum()) or 1.0
        # 정규화된 상대 기여도 (상위 가독성을 위해 반올림)
        return {
            name: round(float(saliency[idx]) / total, 4)
                for idx, name in enumerate(names)
        }

        