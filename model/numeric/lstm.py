""" 
LSTMModel ─ Phase-2 학습 기반 수치 모델

[역할]
- 룰 기반인 ThresholdModel을 대체하는 학습 기반 수치 모델
- 롤링 Z-score 피처(NumericInput.window)로 TripleBarrier 라벨(BUY·HOLD) 예측
- NumericModelPort 계약(run(NumericInput) → NumericSignal) 충족 → 무중단 교체
"""
from __future__ import annotations 

import time 
import numpy as np 
import torch 
from pathlib import Path 
from typing import Optional 

from mps.core.config import cfg, msg 
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
        input_size = cfg.lstm.input_size if input_size is None else input_size 
        hidden_size = cfg.lstm.hidden_size if hidden_size is None else hidden_size
        num_layers = cfg.lstm.num_layers if num_layers is None else num_layers
        num_classes = cfg.lstm.num_classes if num_classes is None else num_classes
        dropout = cfg.lstm.dropout if dropout is None else dropout 

        self.net = torch.nn.LSTM(
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
        out, _ = self.net(x)        # [B, N, hidden]
        last = out[:, -1, :]        # 마지막 타임스탬프
        return self.head(last)      # [B, num_classes(=2)]
    

class LSTMModel:
    """ NumericModelPort 구현체 """
    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None, 
        model_net_params: Optional[dict] = None, 
        attribute: bool = True,
    ) -> None:
        self._device = torch.device(device or cfg.modeling.torch_device)
        self._net_params = model_net_params or cfg.lstm.to_dict()
        self._model = LSTMNet(**self._net_params).to(self._device)
        self._attribute = attribute 
        self._trained = False 

        if weights_path is not None and Path(weights_path).exists():
            ckpt = torch.load(
                weights_path, map_location=self._device, weights_only=True
            )
            state = ckpt[cfg.key.state_dict] \
                if cfg.key.state_dict in ckpt else ckpt
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

        # 1차: no_grad를 통한 빠른 추론으로 방향·신뢰도 결정
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=-1)[0]
            class_idx = int(torch.argmax(probs).item())
            confidence = float(probs[class_idx].item())

        direction: SignalDirection = cfg.data.idx2dir[class_idx]
        if direction == cfg.str.hold:
            return cfg.signal.numeric.no_signal
        
        # 2차: BUY 신호 봉에서만 어트리뷰션(backward 1회 → 백테스트 속도 회복)
        contribution: dict = {}
        if self._attribute:
            x.requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                logits = self._model(x)
            self._model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
            assert x.grad is not None
            saliency = (x.grad[0].abs() * x[0].abs()).sum(dim=0)    # grad X input
            contribution = LSTMModel.build_contrib(
                saliency.detach().cpu().numpy()
            )

        return direction, round(confidence, 4), contribution
    
    def run(self, inp: NumericInput) -> NumericSignal:
        start_time = time.perf_counter()
        direction, confidence, contribution = self.predict(inp)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return NumericSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            direction=direction,
            confidence=confidence,
            feature_contrib=contribution,
            latency_ms=latency_ms
        )
    
    @staticmethod
    def build_contrib(saliency: np.ndarray) -> dict:
        """ 피처별 상대 기여도 할당 """
        total = float(saliency.sum()) or 1.0
        return {
            name: round(float(saliency[idx]) / total, 4)
            for idx, name in enumerate(cfg.modeling.feature_names)
        }
    
    @classmethod
    def from_net(
        cls, 
        net: LSTMNet, 
        device: str = cfg.modeling.torch_device
    ) -> "LSTMModel":
        """ 학습 완료된 네트워크를 추론용으로 제공하는 공식 경로 """
        model = cls(device=device)
        model._model.load_state_dict(net.state_dict())
        model._model.eval()
        model._trained = True 
        return model
