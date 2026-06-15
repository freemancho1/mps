""" 
LSTMModel ─ Phase-2 학습 기반 수치 모델

[역할]
  - 룰 기반의 ThresholdModel을 대체하는 학습 기반 수치 모델.
  - 롤링 Z-score 피처(NumericInput.window)로 TripleBarrier 라벨(BUY·HOLD) 예측
  - NumericModelPort 계약(run(NumericInput) → NumericSignal) 충족 → 무중단 교체
"""
from __future__ import annotations 

import time 
import numpy as np 
import torch 
from pathlib import Path 
from typing import Optional 

from mps.config import cfg, msg 
from mps.core.types import SignalDirection, NumericInput, NumericSignal


class LSTMNet(torch.nn.Module):
    """ 
    시계열 분류 LSTM 모델
    - 입력 [B, N, F=14] → LSTM(마지막 hidden state) → FC head → [B, 2(BUY·HOLD)]
      · B(Batch Size): 한번에 처리할 데이터 양
      · N(Sequence Length or Timesteps): 모델에 입력할 데이터의 크기
      · F(Features): 처리할 속성 수
    """
    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_classes: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        input_size = input_size or cfg.train.lstm_settings.input_size
        hidden_size = hidden_size or cfg.train.lstm_settings.hidden_size
        num_layers = num_layers or cfg.train.lstm_settings.num_layers
        num_classes = num_classes or cfg.train.lstm_settings.num_classes
        if dropout is None:     # dropout은 0이 올 수 있기 때문에 None 비교
            dropout = cfg.train.lstm_settings.dropout

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)       # [B, N, hidden]
        last = out[:, -1, :]        # 마지막 타임스텝
        return self.head(last)      # [B, num_classes]


class LSTMModel:
    """ NumericModelPort Phase-2 구현체 """
    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        model_arch: Optional[dict] = None,
        attribute: bool = True
    ) -> None:
        self._device = torch.device(cfg.model.torch_device) \
            if device is None else torch.device(device)
        self._model_arch = model_arch or cfg.train.lstm_settings.to_dict()
        self._model = LSTMNet(**self._model_arch).to(self._device)
        self._attribute = attribute 
        self._trained = False 

        if weights_path is not None and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self._device, weights_only=True)
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
    
    def predict(self, inp: NumericInput) -> tuple[SignalDirection, float, dict]:
        x = torch.from_numpy(np.ascontiguousarray(inp.window)).float()
        x = x.unsqueeze(0).to(self._device)

        # 1차: no_grad 빠른 추론으로 방향·신뢰도 결정
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=-1)[0]
            cls_idx = int(torch.argmax(probs).item())
            conf = float(probs[cls_idx].item())

        direction: SignalDirection = cfg.data.idx2dir[cls_idx]
        if direction == cfg.str.hold:
            return cfg.trade.signal.no_signal_numeric 
        
        # 2차: BUY 신호 봉에서만 어트리뷰션 (backward 1회 → 백테스트 속도 확보)
        contrib: dict = {}
        if self._attribute:
            x.requires_grad_(True)
            logits = self._model(x)
            self._model.zero_grad(set_to_none=True)
            logits[0, cls_idx].backward()
            assert x.grad is not None, msg.training.not_compute_gradient
            saliency = (x.grad[0].abs() * x[0].abs()).sum(dim=0)    # grad X input
            contrib = LSTMModel.build_contrib(saliency.detach().cpu().numpy())

        return direction, round(conf, 4), contrib
    
    def run(self, inp: NumericInput) -> NumericSignal: 
        """ 추론 시간을 측정해 NumericSignal 생성 (LatencyFilter 근거) """
        start_time = time.perf_counter()
        direction, confidence, contrib = self.predict(inp)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return NumericSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            dir=direction,
            confidence=confidence,
            feature_contrib=contrib,
            latency_ms=latency_ms 
        )
    
    @staticmethod
    def build_contrib(saliency: np.ndarray) -> dict:
        """ 피처별 정규화 상대 기여도 (관측 가능성 원칙 → signals.jsonl 기록) """
        names = cfg.model.feature_names
        total = float(saliency.sum()) or 1.0
        return {
            name: round(float(saliency[idx]) / total, 4)
            for idx, name in enumerate(names)
        }

    @classmethod 
    def from_net(cls, net: LSTMNet, device: str = cfg.model.torch_device) -> "LSTMModel":
        """ 학습 완료 네트워크를 추론 어댑터로 감싸는 공식 경로 (walk-forward용) """
        model = cls(device=device)
        model._model.load_state_dict(net.state_dict())
        model._model.eval()
        model._trained = True 
        return model     

