""" 
FeatureExtractor ─ OHLCV Bar list → 기술적 지표 피처(14개) 행렬(numpy) 생성

[14개 피처]
  - 모멘텀: rsi_14
  - 추세: macd, macd_signal, macd_diff
  - 변동성: bb_upper, bb_mid, bb_lower, bb_pband, atr_14
  - 거래량: obv, volume_ratio
  - 수익률: ret_1, ret_5, ret_20
  
[라이브러리]
  - 기본은 'ta' 라이브러리 이용, 없으면 직접 구현(직접구현 안함)

[이 행렬이 어디에서 사용되나?]
  - extractor.extract(bars) → NumericNormalizer.transform(bars, matrix)
    → Z-score 정규화 후 NumericInput으로 감싸져 ThresholdModel에 전달.
"""
from __future__ import annotations

import numpy as np 
import pandas as pd 

from ta.momentum import RSIIndicator 
from ta.trend import MACD 
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from mps.core.types import Bar 
from mps.config import cfg


class FeatureExtractor:
    """ Bar 리스트 → numpy 피처 행렬 변환 """
    
    def extract(self, bars: list[Bar]) -> np.ndarray:
        """ 
        Bar 리스트 → shape [len(bars), 14] float32 행렬
        - NaN은 0.0으로 채워짐. (초기 봉에서 지표 계산 전 NaN 발생함)
        """
        df = self._to_df(bars)
        features_df = self._compute(df)
        return features_df.values.astype(np.float32)
        
    def _to_df(self, bars: list[Bar]) -> pd.DataFrame:
        """ Bar 리스트 → 데이타프레임 변환 """
        return pd.DataFrame({
            cfg.key.open: [bar.open for bar in bars],
            cfg.key.high: [bar.high for bar in bars],
            cfg.key.low: [bar.low for bar in bars],
            cfg.key.close: [bar.close for bar in bars],
            cfg.key.volume: [float(bar.volume) for bar in bars],    # int → float32
        })
        
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 14개 기술지표 계산 (시가(open)은 사용하지 않음) """
        close, high, low, volume = \
            df[cfg.key.close], df[cfg.key.high], df[cfg.key.low], df[cfg.key.volume]
            
        rsi = RSIIndicator(close, window=14).rsi()
        
        macd_object = MACD(close)
        macd = macd_object.macd()               # MACD선 = EMA(12) - EMA(26)
        macd_signal = macd_object.macd_signal() # 시그널 선 = MACD.EMA(9)
        macd_diff = macd_object.macd_diff()     # 히스토그램 = MACD - Signal
        
        bb = BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()         # 상단 밴드 = 20MA + 2σ
        bb_mid = bb.bollinger_mavg()            # 중간 밴드 = 20MA
        bb_lower = bb.bollinger_lband()         # 하단 밴드 = 20MA - 2σ
        bb_pband = bb.bollinger_pband()
        
        obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        
        volume_ratio = volume / (volume.rolling(20).mean() + cfg.run.zero)
        ret_1 = close.pct_change(1)
        ret_5 = close.pct_change(5)
        ret_20 = close.pct_change(20)
        
        result = pd.DataFrame({
            cfg.key.rsi_14          : rsi,
            cfg.key.macd            : macd,
            cfg.key.macd_signal     : macd_signal,
            cfg.key.macd_diff       : macd_diff,
            cfg.key.bb_upper        : bb_upper,
            cfg.key.bb_mid          : bb_mid,
            cfg.key.bb_lower        : bb_lower,
            cfg.key.bb_pband        : bb_pband,
            cfg.key.obv             : obv,
            cfg.key.atr_14          : atr,
            cfg.key.volume_ratio    : volume_ratio,
            cfg.key.ret_1           : ret_1,
            cfg.key.ret_5           : ret_5,
            cfg.key.ret_20          : ret_20,
        })
        
        # 초반 봉(지표 계산 불가 구간)의 NaN → 0.0으로 채움
        return result.fillna(0.0)
        