""" 
TripleBarrierDataset ─ 학습용 (윈도우, 라벨) 쌍 생성기

[역할]
  - 각 시점 pit(Point-In-Time)의 입력 윈도우 + TripleBarrier 라벨 생성.
  - track="numeric": Z-score 피처 윈도우 [lookback, 14] → LSTM
    track="pattern": 0~1 정규화 OHLCV 윈도우 [lookback, 5] → 1D-CNN

[학습·추론 일관성]
  - 시뮬레이터 추론과 동일한 정규화 공식 사용 (윈도우 단위 Z-score / min-max)
  - 지표는 인과적(causal)이라 전체 시계열에서 한 번 계산 후 슬라이싱 가능.

[데이터 누수 방지]
  - 라벨은 미래 horizon개 봉으로 확정되므로 마지막 horizon개 시점 제외
"""
from __future__ import annotations 

import torch 
import numpy as np 
from typing import Optional 

from mps.core.config import cfg, msg 
from mps.core.types import Bar, TrackType
from mps.data.features import FeatureExtractor, TripleBarrierLabeler
from mps.core.libs import logger 

# TODO 9999-9999: TripleBarrierDataset Test

class TripleBarrierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bars: list[Bar],
        track: Optional[TrackType] = None,
        lookback: Optional[int] = None,
        labeler: Optional[TripleBarrierLabeler] = None,
    ) -> None:
        self._track = cfg.modeling.default_track if track is None else track
        self._lookback = cfg.data.lookback_minutes if lookback is None else lookback 
        self._labeler = TripleBarrierLabeler() if labeler is None else labeler

        labels = self._labeler.label(bars)
        time_horizon = self._labeler.time_horizon
        bar_count = len(bars)

        # 유효구간: 윈도우 충족(pit >= lookback - 1) and 라벨  확정(pit < bar_count - horizon)
        start_pit = self._lookback - 1
        end_pit = bar_count - time_horizon 
        logger.point(msg.feature.ds_window_size(start_pit, end_pit))

        self._X = self._build_numeric_window(bars, start_pit, end_pit) \
                    if track == cfg.modeling.numeric_track else \
                        self._build_pattern_window(bars, start_pit, end_pit)
        
        self._y = labels[start_pit:end_pit].astype(np.int64) \
                    if end_pit > start_pit else np.empty(0, dtype=np.int64)
    
    def _build_numeric_window(self, bars: list[Bar], start: int, end: int) -> np.ndarray:
        """ 수치 트랙 윈도우 생성 """
        features = FeatureExtractor.extract(bars)
        feature_count = features.shape[1]   # 14

        results: list = []
        for pit in range(start, end):
            window = features[pit - self._lookback + 1 : pit + 1]
            mu = window.mean(axis=0)
            std = window.std(axis=0) + cfg.sys.zero 
            results.append(((window - mu) / std).astype(np.float32))

        return (
            np.stack(results) if results else 
            np.empty((0, self._lookback, feature_count), dtype=np.float32)
        )
            
    def _build_pattern_window(self, bars: list[Bar], start: int, end: int) -> np.ndarray:
        """ 패턴 트랙 윈도우 생성 """
        ohlcv = np.array(
            [[bar.open, bar.high, bar.low, bar.close, bar.volume] for bar in bars],
            dtype=np.float64
        )
        feature_count = ohlcv.shape[1]  # 5

        results: list = []
        for pit in range(start, end):
            window = ohlcv[pit - self._lookback + 1 : pit + 1]
            
            prices_window = window[:, :4]
            prices_min, prices_max = prices_window.min(), prices_window.max()
            if prices_max - prices_min < cfg.sys.zero:
                prices_norm = np.zeros_like(prices_window)
            else:
                prices_norm = (prices_window - prices_min) / (prices_max - prices_min)

            volume_window = window[:, 4:5]
            volume_norm = volume_window / (volume_window.max() + cfg.sys.zero)

            window_norm = np.concatenate([prices_norm, volume_norm], axis=1).astype(np.float32)
            results.append(window_norm)

        return (
            np.stack(results) if results else 
            np.empty((0, self._lookback, feature_count), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self._y)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self._X[idx]), int(self._y[idx])
    
    def class_counts(self) -> np.ndarray:
        """ [BUY·HOLD] 클래스별 샘플 수 (불균형 진단·가중치 산정)"""
        return np.bincount(self._y, minlength=cfg.lstm.num_classes)
        

        
