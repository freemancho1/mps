""" 
TripleBarrierDataset ─ 학습용 (윈도우, 라벨) 쌍 생성기

[역할]
  - 봉 시퀀스에서 각 시점 pit(Point-In-Time)의 입력 윈도우와 
    Triple Barrier 라벨을 만들어, PyTorch DataLoader의 입력으로 제공
  - track="numeric": Z-score 정규화 피처 윈도우 [lookback, 14]  → LSTM 학습용
    track="pattern": 0~1 정규화 OHLCV 윈도우 [lookback, 5]      → 1D-CNN 학습용
    
[학습·추론 일관성]
  - 시뮬레이터 추론 시와 동일한 정규화 공식을 사용함.
    · 수치: 윈도우 단위 롤링 Z-score (NumericNormalizer와 동일)
    · 패턴: 윈도우 내 OHLCV min-max, 거래량 max 정규화 (PatternNormalizer와 동일)
  - 지표는 인과적(causal)이라 전체 시계열에서 한 번 계산 후 슬라이싱해도
    각 시점 pit의 값이 추론 시(버퍼 기반)와 동일함.
    
[누출 방지]
  - 라벨은 미래 horizon개의 봉을 봐야 확정되므로, 마지막 horizon개 시점은 제외.
"""
from __future__ import annotations 

import torch 
import numpy as np
from typing import Optional 

from mps.core.types import Bar 
from mps.config import cfg, msg 
from mps.models.numeric.extractor import FeatureExtractor
from mps.pp.features.labeler import TripleBarrierLabeler


class TripleBarrierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bars: list[Bar],
        track: str = cfg.run.numeric_track,         # "numeric"
        lookback: int = cfg.run.lookback_minutes,   # 120
        labeler: Optional[TripleBarrierLabeler] = None,
    ) -> None:
        assert track in (cfg.run.numeric_track, cfg.run.pattern_track)
        self._track = track 
        self._lookback = lookback 
        
        labeler = labeler or TripleBarrierLabeler()
        labels = labeler.label(bars)
        horizon = labeler.time_horizon  # 60분
        num = len(bars)
        
        # 유효 구간: 윈도우가 가득 차고(pit >= lookback - 1) 
        #           라벨이 확정되는 (pit < num - horizon) 시점
        start_pit = self._lookback - 1
        end_pit = num - horizon 
        if track == cfg.run.numeric_track:
            X = self._build_numeric(bars, start_pit, end_pit)
        else:
            X = self._build_pattern(bars, start_pit, end_pit)
            
        self._X = X
        self._y = labels[start_pit:end_pit].astype(np.int64) \
            if end_pit > start_pit else np.empty(0, dtype=np.int64)
            
    # ── 수치 트랙에 대한 데이터셋 생성 ─────────────────
    def _build_numeric(self, bars: list[Bar], start: int, end: int) -> np.ndarray:
        feat = FeatureExtractor().extract(bars)         # [n, 14]
        lookback, zero = self._lookback, cfg.run.zero
        
        out: list = []
        for pit in range(start, end):
            w = feat[pit - lookback + 1 : pit + 1]      # [lookback, 14]
            mu = w.mean(axis=0)
            std = w.std(axis=0) + zero
            out.append(((w - mu) / std).astype(np.float32))
            
        return np.stack(out) if out else np.empty((0, lookback, 14), dtype=np.float32)
    
    # ── 패턴 트랙에 대한 윈도우별 OHLC min-max + 거래량 max 정규화 ──────
    def _build_pattern(self, bars: list[Bar], start: int, end: int) -> np.ndarray: 
        ohlcv = np.array(
            [[bar.open, bar.high, bar.low, bar.close, bar.volume] for bar in bars],
            dtype=np.float64
        )
        lookback, zero = self._lookback, cfg.run.zero
        
        out: list = []
        for pit in range(start, end):
            w = ohlcv[pit - lookback + 1 : pit + 1]
            prices = w[:, :4]
            p_min, p_max = prices.min(), prices.max()
            if p_max - p_min < zero:
                p_norm = np.zeros_like(prices)
            else:
                p_norm = (prices - p_min) / (p_max - p_min)
            v_norm = w[:, 4:5] / (w[:, 4:5].max() + zero)
            out.append(np.concatenate([p_norm, v_norm], axis=1).astype(np.float32))
        
        return np.stack(out) if out else np.empty((0, lookback, 5), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self._y)
    
    def __getitem__(self, idx: int):
        return torch.from_numpy(self._X[idx]), int(self._y[idx])
    
    def class_counts(self) -> np.ndarray:
        """ [BUY·SELL·HOLD] 클래스별 샘플 수 (불균형 진단, 가중치 산정 용) """
        return np.bincount(self._y, minlength=cfg.lstm.num_classes) # 3
        
