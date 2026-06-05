""" 
Phase-1 단계: 수치분석 트랙 모델

[주요 특징]
  - RSI 과매도·과매수 + MACD 크로스오버 조합의 룰 기반 신호 생성
    → 단순하지만 해석이 가능하고 파이프라인 전체가 정상 작동하는지 검증하는데 적합
    
  신호 조건:
  - BUY : RSI < 35 (과매도) AND MACD 골든크로스 (전봉 diff < 0, 현봉 diff >= 0)
  - SELL: RSI > 65 (과매수) AND MACD 데드크로스 (전봉 diff > 0, 현봉 diff <= 0)
  - HOLD: 위 조건 모두 불충족
  
  confidence 계산 (BUY 예):
  - conf = min(1.0, (rsi_low - rsi) / rsi_low + 0.3) ⇒ RSI가 낮을수록 더 강한 확신도
    (기본값인 0.3(closeover_base)은 클로스오버 자체의 베이스 신뢰도임)
    
[교체]
  - Phase-2 이상에서 학습기반 모델인 LSTM·Transformer로 교체 예정
  - ModelPort 인터페이스를 유지하면서 이 파일만 교체하면 가능
"""
from __future__ import annotations 

import time 

from mps.config import cfg 
from mps.core.types import NumericInput, NumericSignal, Direction


class ThresholdModel:
    """ Phase-1 임시 수치 모델 ─ RSI 과매도·과매수 + MACD 클로스오버 비교 """
    
    def __init__(
        self,
        # RSI 과매도 임계값 (이 이하이면 BUY 후보)
        rsi_oversold: float = cfg.run.rsi_oversold,             # 35.0
        # RSI 과매수 임계값 (이 이상이면 SELL 후보)
        rsi_overbought: float = cfg.run.rsi_overbought,         # 65.0
        # RSI 클로스오버 베이스 임계값
        rsi_closeover_base: float = cfg.run.rsi_closeover_base  # 0.3
    ) -> None: 
        self._rsi_lline = rsi_oversold
        self._rsi_hline = rsi_overbought
        self._rsi_base = rsi_closeover_base
        
    def predict(self, inp: NumericInput) -> tuple[Direction, float, dict]:
        """ 
        거래방향, 신뢰도, 피처 기여도를 예측 
        
        [중요 사항]
          - 룰 판정은 inp.raw_window(원본 지표값)를 사용함.
          - inp.window는 Z-score 정규화 값이라 RSI-35/65 임계값과 MACD 히스토그램의
            부호(골드·데드크로스)가 모두 왜곡되므로, 
            절대 기준을 사용하는 ThresholdModel에는 부적합하고, 
            phase-2 이상의 학습용 데이터로 사용함.
        """
        raw = inp.raw_window 
        last = raw[-1]                              # 가장 최근봉
        prev = raw[-2] if len(raw) >= 2 else last   # 가장 최근봉 직전봉
        
        # 주요 피처 추출 (원본 값 ─ RSI는 0~100, MACD diff는 히스토그램 실수값)
        rsi = float(last[cfg.run.feature_idx[cfg.key.rsi_14]])
        macd_diff_now = float(last[cfg.run.feature_idx[cfg.key.macd_diff]])     # 현재 MACD 히스토그램
        macd_diff_prev = float(prev[cfg.run.feature_idx[cfg.key.macd_diff]])    # 직전 MACD 히스토그램
        bb_pband = float(last[cfg.run.feature_idx[cfg.key.bb_pband]])           # 볼린저밴드 위치
        ret_1 = float(last[cfg.run.feature_idx[cfg.key.ret_1]])                 # 직전 1분 수익율
        
        # 관측 가능성 저장 ─ 신호에 기여한 피처 값 기록
        contrib = {
            cfg.key.rsi_14: rsi, 
            cfg.key.macd_diff: macd_diff_now,
            cfg.key.bb_pband: bb_pband,
            cfg.key.ret_1: ret_1
        }
        
        # BUY 조건
        # 골든크로스: 히스토그램이 음수에서 양수로 전환 (MACD선이 시그널 선을 상향 돌파)
        if rsi < self._rsi_lline and macd_diff_prev < 0 and macd_diff_now >= 0:
            conf = min(1.0, (self._rsi_lline - rsi) / self._rsi_lline + self._rsi_base)
            return cfg.key.BUY, round(conf, 4), contrib
        
        # SELL 조건
        if rsi > self._rsi_hline and macd_diff_prev > 0 and macd_diff_now <= 0:
            conf = min(1.0, (rsi - self._rsi_hline) / (100 - self._rsi_hline) + self._rsi_base)
            return cfg.key.SELL, round(conf, 4), contrib
        
        return cfg.run.no_signal
    
    def run(self, inp: NumericInput) -> NumericSignal:
        """ 
        추론 시간을 측정하여 NumericSignal을 생성하고 반환.
        - latency_ms는 LatencyGuard에서 총 지연시간 계산에 사용됨.
        """
        curr_time = time.perf_counter()
        direction, confidence, contrib = self.predict(inp)
        latency_ms = (time.perf_counter() - curr_time) * 1000
        
        return NumericSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            direction=direction,
            confidence=confidence,
            feature_contrib=contrib,
            latency_ms=latency_ms
        )