""" 
정규화 레이어 ─ 수치 트랙(롤링 Z-score)과 패턴 트랙(0~1 상대 정규화) 정규화

[왜 두 가지 정규화를 쓰는가?]
  - 수치 트랙(NumericNormalizer)
    RSI, MACD, 볼린저밴드 등 기술 지표의 "지금이 평소와 얼마나 다른가?"를 표현
    → 롤링 Z-score: (현재값 - 최근 평균) / 최근 표준편차
    → 결과값: 0 = 평균, +2 = 2분위 위 비정상 상승, -2 = 2분위 아래 비정상 하락

  - 패턴 트랙(PatternNormalizer)
    캔들 패턴은 절대 가격 수준과 무관해야 함 (20만원 주식의 망치형 = 5만원 주식의 망치형)
    → 0~1 Min-Max 정규화: 윈도우 내 최저가를 0, 최고가를 1로 매핑
    → 동일 패턴이 가격대에 무관하게 같은 특징 벡터를 가지게 됨.
"""
from __future__ import annotations

import numpy as np 
import pandas as pd 
from typing import Optional 

from mps.config import cfg, msg 
from mps.core.types import Bar, NumericInput, PatternInput


class NumericNormalizer:
    """ 
    롤링 Z-score 정규화.

    입력: Bar 리스트 + FeatureExtractor가 계산한 raw feature_matrix ([N, 14])
    출력: NumericInput (window: 마지막 lookback 행만 추출 후 Z-score 적용)

    [Z-score 계산 방식]
      mu = window 전체 행의 컬럼별 평균
      std = window 전체 행의 컬럼별 표준 편차(+ cfg.run.zero)
      normalized = (window - mu) / std
    """
    def __init__(self, window_size: Optional[int] = None) -> None:
        # window_size = 0으로 들어와도 lookback_minutes로 변경됨
        self._window_size = window_size or cfg.run.lookback_minutes

    def transform(self, bars: list[Bar], feature_matrix: np.ndarray) -> NumericInput:
        """ 
        feature_matrix: shape[len(bars), num_features] ─ FeatureExtractor 결과

        마지막 self._window_size개 행만 잘라서 Z-score 적용.
        → 롤링 정규화이므로 최근 기간만 기준으로 사용함.
        """
        if len(bars) < self._window_size:
            raise ValueError(msg.trade.bt.normal_skip_err(bars, self._window_size))
        
        window = feature_matrix[-self._window_size:]    # 최근 lookback 행 추출
        mean = window.mean(axis=0)
        std = window.std(axis=0) + cfg.run.zero
        normalized = (window - mean) / std

        curr_bar = bars[-1]
        return NumericInput(
            window=normalized.astype(np.float32),
            raw_window=window.astype(np.float32),       # 정규화 이전 원본 (롤 모델 임계값 판정용)
            window_size=self._window_size,
            ticker=curr_bar.ticker,
            timestamp=curr_bar.timestamp
        )
    

class PatternNormalizer:
    """ 
    윈도우 낸 최저가~최고가를 0~1로 매핑하는 상대 정규화

    입력: Bar 리슽 (버퍼에서 마지막 lookback개 봉 사용)
    출력: PatternInput(ohlcv_series: OHLC는 0~1, V는 max기준 0~1)

    [가격 정규화]
      - p_min = 윈도우 내 모든 OHLC 값 중 최소값
      - p_max =   ..                     최대값
      - price_norm = (price - p_min) / (p_max - p_min)
        → 값: 0 = 윈도우 내 최저점, 1 = 윈도우 내 최고점

    [거래량 정규화]
      - volume_norm = volume / max_volume_in_window
        → 가격과 거래량은 단위가 원천적으로 달라 별도 정규화 수행
    """
    def __init__(self, window_size: Optional[int] = None) -> None:
        self._window_size = window_size or cfg.run.lookback_minutes     # 120

    def transform(self, bars: list[Bar]) -> PatternInput:
        """ Bar 리스트에서 마지막 lookback봉을 OHLCV 행렬로 변환 후 정규화 """
        if len(bars) < self._window_size:
            raise ValueError(msg.trade.bt.normal_skip_err(bars, self._window_size))
        
        window = bars[-self._window_size:]
        ohlcv = np.array(
            [[b.open, b.high, b.low, b.close, b.volume] for b in window],
            dtype=np.float64
        )

        # 가격 4개열
        price_columns = ohlcv[:, :4]
        p_min, p_max = price_columns.min(), price_columns.max()
        # 가격 변동이 극단적으로 적은 경우 오차가 큰 값이 나올 수 있으므로 0으로 변환
        if p_max - p_min < cfg.run.zero:
            price_normalized = np.zeros_like(price_columns)
        else:
            price_normalized = (price_columns - p_min) / (p_max - p_min)

        # 거래량
        volume_columns = ohlcv[:, 4:5]
        volume_max = volume_columns.max()
        volume_normalized = volume_columns / (volume_max + cfg.run.zero)

        normalized = np.concatenate([price_normalized, volume_normalized], axis=1) \
                       .astype(np.float32)
        
        curr_bar = bars[-1]
        return PatternInput(
            ohlcv_series=normalized,
            ticker=curr_bar.ticker,
            timestamp=curr_bar.timestamp
        )