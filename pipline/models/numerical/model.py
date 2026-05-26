""" 
수치 분석 트랙 모델

[Phase-1: ThresholdModel]
  - RSI 과매도·과매수 + MACD 크로스오버 조합의 룰 기반 신호 생성
    → 단순하지만 해석이 가능하고 파이프라인 전체가 정상 작동하는지 검증하는데 적합

[교체 계획]
  - Phase-2: 학습 기반 모델(LSTM / Transformer)로 교체 예정
  - ModelPort 인터페이스를 유지하면서 이 파일만 교체하면 가능

[신호 조건]
  - BUY : RSI < 35 (과매도) AND MACD 골든크로스 (전봉 diff < 0, 현봉 diff >= 0)
  - SELL: RSI > 65 (과매수) AND MACD 데드크로스 (전봉 diff > 0, 현봉 diff <= 0)
  - HOLD: 위 조건 불충족

  - confidence 계산 (BUY 예):
    conf = min(1.0, (rsi_low - rsi) / rsi_low + 0.3)
    RSI가 낮을수록 더 강한 확신도.
    (기본값 0.3은 크로스오버 자체의 베이스 신뢰도)
"""
from __future__ import annotations 

import time 
import numpy as np 

from mps.pipline.models.numerical.extractor import FeatureExtractor
from mps.sys.core.types import NumericalInput, NumericalSignal, Direction
from mps.sys import cfg

# FeatureExtractor.FEATURE_NAMES 순서에서 각 피처의 열 인덱스를 미리 매핑해
# ThresholdModel내에서 w[pos][_IDX["rsi_14"]]와 같이 접근하기 위함
_IDX = {name: idx for idx, name in enumerate(FeatureExtractor.FEATURE_NAMES)}


class ThresholdModel:
    """ Phase-1 임시 수치 모델 ─ RSI 과매도·과매수 + MACD 크로스오버 비교 """

    def __init__(
        self,
        # RSI 과매도 임계값 (이 이하 → BUY 후보)
        rsi_oversold: float = cfg.sys.rsi_oversold,     
        # RSI 과매수 임계값 (이 이상 → SELL 후보)
        rsi_overbought: float = cfg.sys.rsi_overbought,
        # RSI 크로스오버 베이스 임계값
        rsi_closeover_base: float = cfg.sys.rsi_closeover_base,
    ) -> None:
        self._rsi_low = rsi_oversold
        self._rsi_high = rsi_overbought 
        self._rsi_base = rsi_closeover_base

    def predict(self, inp: NumericalInput) -> tuple[Direction, float, dict]:
        """ 
        방향, 신뢰도, 피처 기여도 튜플 반환

        inp.window는 Z-score 정규화된 값이므로 절대적인 RSI 35 기준이 아닌 상대적 위치
        (NumericalNormalizer가 정규화했으므로 실제 RSI 원본 값이 아님)
        """
        window = inp.window                             # shape = [N, 14] ─ Z-score 정규화 피처
        last = window[-1]                               # 가장 최근 봉
        prev = window[-2] if len(window) >=2 else last  # 가장 최근 봉 직전 봉

        # 주요 피처 추출
        rsi = float(last(_IDX["rsi_14"]))               
        macd_diff_now = float(last[_IDX["macd_diff"]])  # 현재 MACD 히스토그램
        macd_diff_prev = float(prev[_IDX["macd_diff"]]) # 직전 MACD 히스토그램
        bb_pband = float(last[_IDX["bb_pband"]])        # 볼린저밴드 위치
        ret_1 = float(last[_IDX["ret_1"]])              # 직전 1분 수익률

        # 관측 가능성: 신호에 기여한 피처 값을 항상 기록
        contrib = {
            "rsi_14": rsi, 
            "macd_diff": macd_diff_now,
            "bb_pband": bb_pband,
            "ret_1": ret_1
        }

        # ── BUY 조건: RSI 과매도 + MACD 골든크로스 ──────────────
        # 골든크로스: 히스토그램이 음수에서 양수로 전환 (MACD 선이 시그널 선을 상향 돌파)
        if rsi < self._rsi_low and macd_diff_prev < 0 and macd_diff_now >= 0:
            conf = min(1.0, (self._rsi_low - rsi) / self._rsi_low + self._rsi_base)
            return "BUY", round(conf, 4), contrib 
        
        # ── SELL 조건: RSI 과매수 + MACD 데드크로스 ──────────────
        if rsi > self._rsi_high and macd_diff_prev > 0 and macd_diff_now <= 0:
            conf = min(1.0, (rsi - self._rsi_high) / (100 - self._rsi_high) + self._rsi_base)
            return "SELL", round(conf, 4), contrib 
        
        return "HOLD", 0.0, contrib 
    
    def run(self, inp: NumericalInput) -> NumericalSignal:
        """ 
        추론 시간을 측정하여 NumericalSignal을 생성하고 반환
        - latency_ms는 LatencyGuard에서 총 지연시간 계산에 사용됨
        """
        curr_time = time.perf_counter()
        direction, confidence, contrib = self.predict(inp)
        latency_ms = (time.perf_counter() - curr_time) * 1000
        
        return NumericalSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            direction=direction,
            confidence=confidence,
            feature_contrib=contrib,
            latency_ms=latency_ms
        )