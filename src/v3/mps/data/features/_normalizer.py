""" 
정규화 레이어 ─ 수치(롤링 Z-score)·패턴(0~1 상대 정규화) 트랙

[왜 두 가지 정규화를 수행하나?]
  - 수치 트랙(NumericNormalizer): '지금이 평소 대비 얼마나 다른가?'
    → 롤링 Z-score: (현재값 - 최근 평균) / 최근 표준 편차
  - 패턴 트랙(PatternNormalizer): 캔들 패턴은 절대 가격과 무관해야 함.
    → 윈도우 내 최저가=0, 최고가=1 매핑, 거래량은 원도우 max 기준 0~1
"""
from __future__ import annotations 

import numpy as np 
from typing import Optional 

from mps.config import cfg, msg
from mps.core.types import Bar, NumericInput, PatternInput 


class NumericNormalizer:
    """ 
    수치 트랙 정규화 ─ 롤링 Z-score 
    
    입력: Bar 리스트 + FeatureExtractor의 raw 피처 행렬 [N, 14]
    출력: NumericInput (window: 마지막 lookback 행 Z-score / raw_window: 원본)
    """
    def __init__(self, window_size: Optional[int] = None) -> None:
        self._window_size = window_size or cfg.data.lookback_minutes

    def transform(self, bars: list[Bar], feature_matrix: np.ndarray) -> NumericInput:
        if len(bars) < self._window_size:
            raise ValueError(msg.pp.features.normal_size_err(bars, self._window_size))
        
        window = feature_matrix[-self._window_size:]    # 최근 lookback 개
        mean = window.mean(axis=0)
        std = window.std(axis=0) + cfg.sys.zero
        normalized = (window - mean) / std 

        curr_bar = bars[-1]
        return NumericInput(
            window=normalized.astype(np.float32),
            raw_window=window.astype(np.float32),       # 롤 모델 임계값 판정용 원본 별도 저장
            window_size=self._window_size,
            ticker=curr_bar.ticker,
            timestamp=curr_bar.timestamp 
        )
    

class PatternNormalizer:
    """ 
    윈도우 내 최저~최고가를 0~1로 매핑하는 상대 정규화.

    [가격]   price_norm = (price - p_min) / (p_max - p_min)
    [거래량] volume_norm = volume / max_volume (단위가 달라 별도 정규화)
    """
    def __init__(self, window_size: Optional[int] = None) -> None:
        self._window_size = window_size or cfg.data.lookback_minutes

    def transform(self, bars: list[Bar]) -> PatternInput:
        if len(bars) < self._window_size:
            raise ValueError(msg.pp.features.normal_size_err(bars, self._window_size))
        
        ohlcv = np.array(
            [[bar.open, bar.high, bar.low, bar.close, bar.volume]
                for bar in bars[-self._window_size:]]
        )

        price_columns = ohlcv[:, :4]
        p_min, p_max = price_columns.min(), price_columns.max()
        if p_max - p_min < cfg.sys.zero:        # 가격 변동성이 사실상 0인 구간 보호
            price_normalized = np.zeros_like(price_columns)
        else:
            price_normalized = (price_columns - p_min) / (p_max - p_min)

        volume_columns = ohlcv[:, 4:5]
        volume_normalized = volume_columns / (volume_columns.max() + cfg.sys.zero)

        normalized = np.concatenate(
            [price_normalized, volume_normalized], axis=1
        ).astype(np.float32)

        curr_bar = bars[-1]
        return PatternInput(
            ohlcv_series=normalized,
            ticker=curr_bar.ticker,
            timestamp=curr_bar.timestamp
        )