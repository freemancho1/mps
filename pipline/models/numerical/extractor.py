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
from mps.sys import cfg, msg

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
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
        features_df = self._compute(df)
        # print(msg.hs.extract_result(bars, df, features_df))
        return features_df.values.astype(np.float32)
    
    def _to_df(self, bars: list[Bar]) -> pd.DataFrame:
        """ Bar 리스트 → pd.DataFrame 변환 """
        return pd.DataFrame({
            "open": [bar.open for bar in bars],
            "high": [bar.high for bar in bars],
            "low":  [bar.low for bar in bars],
            "close": [bar.close for bar in bars],
            "volume": [float(bar.volume) for bar in bars],  # int → float32
        })
    
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 14개 기술 지표 계산 """
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

        if _TA_AVAILABLE:
            # ── ta 라이브러리 사용 (권장) ─────────────────
            rsi = RSIIndicator(close, window=14).rsi()

            macd_obj = MACD(close)
            macd = macd_obj.macd()              # MACD 선 = EMA(12) - EMA(26)
            macd_sig = macd_obj.macd_signal()   # 시그널 선 = MACD.EMA(9)
            macd_diff = macd_obj.macd_diff()    # 히스토그램 = MACD - Signal

            bb = BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()     # 상단 밴드 = 20MA + 2σ
            bb_mid = bb.bollinger_mavg()        # 중간 밴드 = 20MA
            bb_lower = bb.bollinger_lband()     # 하단 밴드 = 20MA - 2σ
            # pband: (close - lower) / (upper - lower) → 0=하단, 0.5=중간, 1=상단
            bb_pband = bb.bollinger_pband()

            obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        else:
            # ── fallback: ta 라이브러리가 없을 때 직접 계산 ────────
            rsi = self._rsi(close, 14)
            macd, macd_sig, macd_diff = self._macd(close)
            bb_upper, bb_mid, bb_lower = self._bollinger(close)
            bb_pband = (close - bb_lower) / (bb_upper - bb_lower + cfg.sys.zero)
            obv = (volume * np.sign(close.diff())).cumsum()
            atr = (high - low).rolling(14).mean()

        # ── 공통 계산 구간 ────────────────────────    
        vol_ratio = volume / (volume.rolling(20).mean() + cfg.sys.zero)
        ret_1 = close.pct_change(1)
        ret_5 = close.pct_change(5)
        ret_20 = close.pct_change(20)

        result = pd.DataFrame({
            "rsi_14": rsi,
            "macd": macd,
            "macd_signal": macd_sig,
            "macd_diff": macd_diff,
            "bb_upper": bb_upper,
            "bb_mid": bb_mid,
            "bb_lower": bb_lower,
            "bb_pband": bb_pband,
            "obv": obv,
            "atr_14": atr,
            "volume_ratio": vol_ratio,
            "ret_1": ret_1,
            "ret_5": ret_5,
            "ret_20": ret_20,
        })
        # 초반 봉(지표 계산 불가 구간)의 NaN → 0.0 채움
        return result.fillna(0.0)

    # ── fallback 구현 ───────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        """Wilder 방식 RSI.
        gain/loss 를 rolling mean 으로 계산 (ta 와 동일 결과).
        """
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """표준 MACD: EMA(12) - EMA(26), 시그널 EMA(9)."""
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal, macd - signal

    @staticmethod
    def _bollinger(close: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저밴드: 20MA ± 2σ."""
        mid = close.rolling(window).mean()
        std = close.rolling(window).std()
        return mid + 2 * std, mid, mid - 2 * std