""" 
FeatureExtractor ─ OHLCV Bar 리스트 → 기술적 지표 피처(14개) 행렬(numpy) 생성

[14개 피처]
  - 모멘텀: rsi_14
  - 추세:   macd, macd_signal, macd_diff
  - 변동성: bb_upper, bb_mid, bb_lower, bb_pband, atr_14
  - 거래량: obv, volume_ratio
  - 수익률: ret_1, ret_5, ret_20
  
[라이브러리]
  - 기본은 'ta' 라이브러리를 이용하고, 없으면 직접구현.
  - 두 구현이 동일한 결과를 내도록 설계되어 있으나, production에서는 ta 설치 권장
  
[이 행렬이 어디로 가는가?]
  - extractor.extract(bars) → NumericalNormalizer.transform(bars, matrix)
    → Z-score 정규화 후 NumericalInput으로 감싸져 ThresholdModel에 전달.
"""
from __future__ import annotations 

import numpy as np 
import pandas as pd

from mps.sys.core.types import Bar 
from mps.sys import msg

try:
    import ta 
    _TA_AVAILABLE = True 
except ImportError:
    _TA_AVAILABLE = False 
    

class FeatureExtractor:
    """ Bar 리스트 → numpy 피처 행렬 변환. """
    # 피처 순서 고정
    FEATURE_NAMES = [
        "rsi_14",
        "macd", "macd_signal", "macd_diff",
        "bb_upper", "bb_mid", "bb_lower",
        "bb_pband",
        "obv",
        "atr_14",
        "volume_ratio",
        "ret_1", "ret_5", "ret_20"
    ]
    
    def extract(self, bars: list[Bar]) -> np.ndarray:
        """ 
        Bar 리스트 → shape [len(bars), 14] float32 행렬

        - NaN은 0.0으로 채워진다. (초기 봉에서 지표 계산 전 NaN 발생)
        """
        df = self._to_df(bars)
        features = self._compute(df)
        print(msg.hs.extract_result(bars, df, features))
        return features.values.astype(np.float32)
    
    def _to_df(self, bars: list[Bar]):
        pass
    
    def _compute(self, df: pd.DataFrame):
        pass