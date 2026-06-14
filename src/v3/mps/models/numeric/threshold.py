""" 
Phase-1 단계: 경계를 기준으로 수치 데이터를 분석하는 수치분석 모델

[주요 특징]
  - RSI 과매도·과매수 + MACD 크로스오버 조합의 룰 기반 신호 생성
    → 단순하지만 해석이 가능하고 파이프라인 전체가 정상 작동하는지 검증하는데 적합함.

[신호 조건]
  - BUY : RSI < 35 (과매도) and MACD 골든크로스 (전봉 diff < 0, 현봉 diff >= 0)
  - SELL: RSI > 65 (과매수) and MACD 데드크로스 (전봉 diff > 0, 현봉 diff <= 0)
  - HOLD: 위 조건 모두 불충족

[confidence 계산]
  - confidence = min(1.0, (rsi_low - rsi) / rsi_low + 0.3) ⇒ RSI가 낮을수록 더 강한 확신도
    (기본값인 0.3(closeover_base)은 클로스오버 자체의 베이스 신뢰도임)

[교체 계획]
  - Phase-2 이상에서 학습 기반 모델인 LSTM·Transformer로 교체 예정
  - ModelPort 인터페이스를 유지하면서 이 파일만 교체하면 가능 함
"""
from __future__ import annotations 

import time 
from typing import Optional 

from mps.config import cfg 
from mps.core.types import NumericInput, NumericSignal, SignalDirection


class ThresholdModel:
    """ Phase-1 임수 수치 모델 ─ RSI 과매도·과매수 + MACD 클로스오버 비교 """

    def __init__(
        self,
        rsi_oversold: Optional[float] = None,
        rsi_overbought: Optional[float] = None,
        rsi_closeover_base: Optional[float] = None,
    ) -> None:
        self._rsi_oversold = cfg.trade.signal.rsi_oversold \
            if rsi_oversold is None else rsi_oversold
        self._rsi_overbought = cfg.trade.signal.rsi_overbought \
            if rsi_overbought is None else rsi_overbought 
        self._closeover_base = cfg.trade.signal.rsi_closeover_base \
            if rsi_closeover_base is None else rsi_closeover_base
        
    def predict(self, inp: NumericInput) -> tuple[SignalDirection, float, dict]:
        """ 
        거래방향, 신뢰도 및 피처 기여도를 예측

        [주요 기능]
          - 룰 판정은 inp.raw_window(원본 지표값)를 사용함.
          - inp.window는 Z-score 정규화 값으로 RSI-35·65 임계값과 MACD 히스토그램의
            부호(골드·데드크로스)가 모두 왜곡되므로,
            절대 기준을 사용하는 ThresholdModel에서는 부적합함(Phase-2부터 사용할 예정)
        """
        window = inp.raw_window 
        last = window[-1]                                   # 가장 최근봉
        prev = window[-2] if len(window) >= 2 else last     # 가장 최근봉 직전봉

        # 주요 피처 추출 (원본값 ─ RSI는 0~100, MACD diff는 히스토그램 실수값)
        rsi = float(last[cfg.model.feature_idx[cfg.key.rsi_14]])
        macd_diff_now = float(last[cfg.model.feature_idx[cfg.key.macd_diff]])
        macd_diff_prev = float(prev[cfg.model.feature_idx[cfg.key.macd_diff]])
        bb_pband = float(last[cfg.model.feature_idx[cfg.key.bb_pband]])
        ret_1 = float(last[cfg.model.feature_idx[cfg.key.ret_1]])

        # 관측 가능성 저장 ─ 신호에 기여한 피처 값 기록
        contrib = {
            cfg.key.rsi_14: rsi,
            cfg.key.macd_diff: macd_diff_now,
            cfg.key.bb_pband: bb_pband,
            cfg.key.ret_1: ret_1
        }

        # BUY 조건
        # 골든크로스: 히스토그램이 음수에서 양수로 전환 (MACD선이 시그널 선을 상향 돌파)
        if rsi < self._rsi_oversold and macd_diff_prev < 0 and macd_diff_now >= 0:
            conf = min(1.0, (self._rsi_oversold - rsi) / self._rsi_oversold + self._closeover_base)
            return cfg.str.buy, round(conf, 4), contrib
        
        return cfg.trade.signal.no_signal_numeric
    
    def run(self, inp: NumericInput) -> NumericSignal:
        """ 
        추론 시간을 측정하여 NumericSignal을 생성하고 반환.
        - latency_ms는 LatencyGuard에서 총 지연시간 계산에 사용됨.
        """
        start_time = time.perf_counter()
        direction, confidence, contrib = self.predict(inp)
        latency_ms = (time.perf_counter() - start_time) * 1000 

        return NumericSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            dir=direction,
            confidence=confidence,
            feature_contrib=contrib,
            latency_ms=latency_ms
        )

