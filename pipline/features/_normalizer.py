""" 
정규화 레이어 ─ 수치 트랙(롤링 Z-score)과 패턴 트랙(0~1 상대 정규화) 정규화

[왜 두 가지 정규화를 쓰는가?]
  수치 트랙(NumericalNormalizer)
    - RSI, MACD, 볼린저밴드 등 기술 지표의 "지금이 평소와 얼마나 다른가?"를 표현
      → 롤링 Z-score: (현재값 - 최근 평균) / 최근 표준편차
      → 결과값 0은 평균, +2는 2분위 위 비정상적인 상승, -2는 2분위 아래 비정상적인 하락.

  패턴 트랙(PatternNormalizer)
    - 캔들 패턴은 절대 가격 수준과 무관해야 함 (20만원 주식의 망치형 = 5만원 주식의 망치형)
      → 0~1 Min-Max 정규화: 윈도우 내 최저가를 0, 최고가를 1로 매핑.
      → 동일 패턴이 가격대에 무관하게 같은 특징 벡터를 가지게 됨.
"""
from __future__ import annotations

import numpy as np 
import pandas as pd 
from typing import Optional 

from mps.sys.core.types import Bar, NumericalInput, PatternInput
from mps.sys import cfg, msg 


class NumericalNormalizer:
    """ 
    롤링 Z-score 정규화.

    입력: Bar 리스트 + FeatureExtractor가 계산한 raw feature_matrix (shape [N, 14])
    출력: NumericalInput (window: 마지막 lookback 행만 추출 후 Z-score 적용)

    Z-score 계산 방식:
      mu = window 전체 행의 컬럼별 평균
      std = window 전체 행의 컬럼별 표준편차 + 1e-8 (0 나눔 방지)
      normalized = (window - mu) / std
    """
    def __init__(self, window_size: Optional[int] = None) -> None:
        # lookback_minutes 기본값
        self._window_size = window_size or cfg.sys.lookback_minutes

    def transform(self, bars: list[Bar], feature_matrix: np.ndarray) -> NumericalInput:
        """ 
        feature_matrix: shape [len(bars), num_features] ─ featureExtractor 결과

        마지막 self._window_size개 행만 잘라서 Z-score 적용.
        → 전체를 사용하지 않고 마지막만 쓰는 이유: 롤링 정규화이므로 최근 기간만 기준으로 함.
        """
        if len(bars) < self._window_size:
            raise ValueError(msg.hs.normal.size_err(self._window_size, bars))
        
        window = feature_matrix[-self._window_size:]    # 최근 lookback 행 추출
        mu = window.mean(axis=0)
        std = window.std(axis=0) + cfg.sys.zero
        normalized = (window - mu) / std

        return NumericalInput(
            window=normalized.astype(np.float32),
            window_size=self._window_size,
            ticker=bars[-1].ticker,
            timestamp=bars[-1].timestamp,
        )
    

class PatternNormalizer:
    """ 
    윈도우 내 최저가~최고가를 0~1로 매핑하는 상대 정규화

    입력: Bar리스트 (버퍼에서 마지막 lookback 개 봉 사용)
    출력: PatternInput(ohlcv_series: OHLC는 0~1, V는 max기준 0~1)

    [가격 정규화]
      - p_min = 윈도우 내 모든 OHLC 값의 최소값
      - p_max = 윈도우 내 모든 OHLC 값의 최댓값
      - price_norm = (price - p_min) / (p_max - p_min)
        → 값: 0 = 윈도우 내 최저점, 1 = 윈도우 내 최고점
    
    [거래량 정규화]
      - volume_norm = volume / max_volume_in_window
        → OHLC와 별도로 정규화하는 이유: 가격 단위와 거래량 단위가 완전히 다름
    """
    def __init__(self, window_size: Optional[int] = None) -> None:
        self._window_size = window_size or cfg.sys.lookback_minutes

    def transform(self, bars: list[Bar]) -> PatternInput:
        """ Bar 리스트에서 마지막 lookback봉을 OHLCV 행렬로 변환 후 정규화 """
        if len(bars) < self._window_size:
            raise ValueError(msg.hs.normal.size_err(self._window_size, bars))
        
        window_bars = bars[-self._window_size:]
        # shape [lookback, 5] ⇒ 각 행별로 [Open, High, Low, Close, Volume]
        ohlcv = np.array(
            [[bar.open, bar.high, bar.low, bar.close, bar.volume]
             for bar in window_bars],
            dtype=np.float64
        )

        # 가격 4개 열(Open, High, Low, Close)에 대한 0~1(min-max) 정규화
        price_columns = ohlcv[:, :4]
        p_min, p_max = price_columns.min(), price_columns.max()
        # 가격 변동이 1e-9 정도의 오차가 있는 경우 큰 값이 나올 수 있으므로,
        # 이 코드를 사용하는것이 더 안전함
        if p_max - p_min < cfg.sys.zero:
            # 가격 변동이 없는 극단적인 경우는 모든 값을 0으로 변환
            price_norm = np.zeros_like(price_columns)
        else:
            price_norm = (price_columns - p_min) / (p_max - p_min)
        
        # 거래량 열: 윈도우 내 최대 거래량 기준 0~1
        volume_columns = ohlcv[:, 4:5]
        volume_max = volume_columns.max()
        volume_norm = volume_columns / (volume_max + cfg.sys.zero)

        normalized = np.concatenate(
            [price_norm, volume_norm], 
            axis=1
        ).astype(np.float32)

        return PatternInput(
            ohlcv_series=normalized,
            ticker=bars[-1].ticker,
            timestamp=bars[-1].timestamp
        )