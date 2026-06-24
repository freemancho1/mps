""" 정규화 레이어 ─ 수치(롤링 Z-score)·패턴(0~1 정규화) 트랙 정규화 """
from __future__ import annotations 

import numpy as np 
from typing import Optional 

from mps.core.config import cfg, msg 
from mps.core.types import Bar, NumericInput, PatternInput 


class NumericNormalizer:
    """ 
    수치 트랙 정규화 ─ 롤링 Z-score 
    
    #TODO 8888: 최종 출력 shape 확인
    입력: FeatureExtractor의 raw 피처 행렬 [N, 14] 
    출력: NumericInput (window: 마지막 lookback 행 Z-score + raw_window(원본))
    """
    def __init__(self, window_size: Optional[int] = None) -> None:
        self._window_size = cfg.data.lookback_minutes \
            if window_size is None else window_size
        
    # base_bar는 feature_metrix를 만든 마지막 봉
    def transform(
        self, base_bar: Bar, feature_metrix: np.ndarray
    ) -> NumericInput:
        # TODO 8888: transform이 사이즈 오류 났을 때 진행사항 확인
        if len(feature_metrix) < self._window_size:
            raise ValueError(
                msg.feature.too_few_bar_size_err(
                    len(feature_metrix), self._window_size
                )
            )
        
        # 최근 lookback 개로 윈도우 생성
        window = feature_metrix[-self._window_size:]    
        mean = window.mean(axis=0)
        std = window.std(axis=0) + cfg.sys.zero
        normalized = (window - mean) / std 
        
        return NumericInput(
            ticker=base_bar.ticker,
            timestamp=base_bar.timestamp,
            window=normalized.astype(np.float32),
            raw_window=window.astype(np.float32),
            window_size=self._window_size,
        )
        
        
class PatternNormalizer:
    """ 
    윈도우 내 최저~최고가를 0~1로 매핑하는 상대 정규화.
    
    '가격'과 '거래량'은 단위가 달라 별도로 정규화.
    """
    def __init__(self, window_size: Optional[int] = None) -> None:
        self._window_size = cfg.data.lookback_minutes \
            if window_size is None else window_size 
            
    def transform(self, bars: list[Bar]) -> PatternInput:
        # TODO 8888: transform이 사이즈 오류 났을 때 진행사항 확인
        if len(bars) < self._window_size:
            raise ValueError(
                msg.feature.too_few_bar_size_err(len(bars), self._window_size)
            )
            
        ohlcv = np.array([
            [bar.open, bar.high, bar.low, bar.close, bar.volume]
                for bar in bars[-self._window_size:]
        ])
        
        price_columns = ohlcv[:, :4]
        price_min, price_max = price_columns.min(), price_columns.max()
        if price_max - price_min < cfg.sys.zero:    
            # 가격 변동성이 사실상 없는 경우
            price_normalized = np.zeros_like(price_columns)
        else:
            price_normalized = \
                (price_columns - price_min) / (price_max - price_min)
            
        volume_columns = ohlcv[:, 4:5]
        volume_normalized = \
            volume_columns / (volume_columns.max() + cfg.sys.zero)
        
        normalized = np.concatenate(
            [price_normalized, volume_normalized], axis=1
        ).astype(np.float32)
        
        base_bar = bars[-1]
        return PatternInput(
            ticker=base_bar.ticker,
            timestamp=base_bar.timestamp,
            ohlcv_norm=normalized,
        )