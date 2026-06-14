""" 
FeatureExtractor ─ OHLCV Bar 리스트 → 기술지표 피처 14개 생성

[14개 피처 (순서 = cfg.model.feature_names = 행렬 컬럼)]
  - 모멘텀: rsi_14
  - 추세:   macd, macd_signal, macd_diff 
  - 변동성: bb_upper, bb_mid, bb_lower, bb_pband, atr_14
  - 거래량: obv, volume_ratio
  - 수익률: ret_1, ret_5, ret_20

[인과성(causality)이 핵심]
  - 모든 지표가 인과적(현재까지의 봉만 사용: EMA·rolling·pct_change)이므로,
    전체 시계열에서 한 번 계산 후 시점별로 슬라이싱해도 각 시점의 값은
    "그 시점에 버퍼로 계산한 값"과 동일하다.
    → 학습(dataset)·추론(simulator)이 같은 추출기를 공유해도,
       look-ahead가 생기지 않는 근거임

[NaN 처리]
  - 초기 봉(rolling·EMA 워밍업 구간)의 NaN은 0.0으로 채움.
  - 시뮬레이터 버퍼가 lookback + warmup_bars(50)이므로, 
    실제 신호에 쓰이는 마지막 lookback 행에는 NaN→0 잔재가 섞이지 않음.
"""
from __future__ import annotations

import numpy as np 
import pandas as pd 

from ta.momentum import RSIIndicator
from ta.trend import MACD 
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from mps.config import cfg, msg 
from mps.core.types import Bar 


class FeatureExtractor:
    """ Bar 리스트 → numpy 피처 행렬 [N, 14] float """

    def extract(self, bars: list[Bar]) -> np.ndarray:
        df = self._to_df(bars)
        features_df = self._compute(df)
        return features_df.values.astype(np.float32)

    def _to_df(self, bars: list[Bar]) -> pd.DataFrame:
        """ Bar 리스트 → DataFrame (지표 라이브러리 입력 형식) """
        return pd.DataFrame({
            cfg.key.open    : [bar.open for bar in bars],
            cfg.key.high    : [bar.high for bar in bars],
            cfg.key.low     : [bar.low for bar in bars],
            cfg.key.close   : [bar.close for bar in bars],
            cfg.key.volume  : [float(bar.volume) for bar in bars],
        })
    
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 14개 기술지표 계산 (시가는 지표 계산에 사용하지 않음) """
        high, low, close, volume = \
            df[cfg.key.high], df[cfg.key.low], df[cfg.key.close], df[cfg.key.volume]
        
        rsi = RSIIndicator(close, window=14).rsi()

        macd_object = MACD(close)
        macd = macd_object.macd()                   # MACD선 = EMA(12) - EMA(26)
        macd_signal = macd_object.macd_signal()     # 시그널선 = MACD의 EMA(9)
        macd_diff = macd_object.macd_diff()         # 히스토그램 = MACD - Signal

        bb = BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()             # 20MA + 2σ
        bb_mid = bb.bollinger_mavg()                # 20MA
        bb_lower = bb.bollinger_lband()             # 20MA - 2σ
        bb_pband = bb.bollinger_pband()             # 밴드 내 상대 위치 (0~1 부근)

        obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()

        # 거래량 비율: 최근 20봉 평균 대비 현재 거래량 (폭증 감지)
        volume_ratio = volume / (volume.rolling(20).mean() + cfg.sys.zero)
        ret_1 = close.pct_change(1)
        ret_5 = close.pct_change(5)
        ret_20 = close.pct_change(20)

        # 컬럼 순서 = cfg.model.feature_names 순서 (절대 임의 변경 금지)
        result = pd.DataFrame({
            cfg.key.rsi_14: rsi,
            cfg.key.macd: macd,
            cfg.key.macd_signal: macd_signal,
            cfg.key.macd_diff: macd_diff,
            cfg.key.bb_upper: bb_upper,
            cfg.key.bb_mid: bb_mid,
            cfg.key.bb_lower: bb_lower,
            cfg.key.bb_pband: bb_pband,
            cfg.key.obv: obv,
            cfg.key.atr_14: atr,
            cfg.key.volume_ratio: volume_ratio,
            cfg.key.ret_1: ret_1,
            cfg.key.ret_5: ret_5,
            cfg.key.ret_20: ret_20,
        })

        # 워밍업 구간 NaN → 0.0 (워밍업 봉 수 확보로 실사용 윈도우에는 미혼입)
        return result.fillna(0.0)

