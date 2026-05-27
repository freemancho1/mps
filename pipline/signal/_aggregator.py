""" 
SingnalAggregator ─ 수치트랙 신호 + 패턴트랙 신호 결합

[결합 원칙: "두 트랙이 동의해야 진입"]
  - 단일 트랙만으로 진입하면 오신호(false positive)가 많아짐
  - 두 트랙이 서로 다른 방향을 가르키면 시장 신호가 명확하지 않은 것으로 판단(기권)
  
[결합 로직]
  1. 둘 다 HOLD → None (신호 없음)
  2. 방향 반대 → None (합의 실패)
  3. 한쪽만 HOLD → 나머지 트랙의 방향 채택, 점수 = ???_conf * 0.5
  4. 방향 동일 → combined = numeric_conf * 0.5 + pattern_conf * 0.5
  
[교체 계획]
  - Phase-2: 단순 가중 평균 → 메타 모델(스태킹) 또는 강화학습 정책으로 발전 예정
  - 두 트랙의 과거 성과를 기반으로 가중치를 동적 조정하는 방식 검토
"""
from __future__ import annotations 

from typing import Optional

from mps.sys.core.types import BSDirection, NumericalSignal, PatternSignal, TradeSignal
from mps.sys import cfg 


class SignalAggregator:
    def __init__(
        self, 
        numeric_weight: float = cfg.sys.numeric_combined_weight,
        pattern_weight: float = cfg.sys.pattern_combined_weight,
    ) -> None:
        self._numeric_weight = numeric_weight 
        self._pattern_weight = pattern_weight 
        
    def combine(
        self,
        numeric_signal: NumericalSignal,
        pattern_signal: PatternSignal
    ) -> Optional[TradeSignal]:
        """ 
        두 트랙의 신호를 결합하여 TradeSignal 또는 None 반환
        
        combined_score가 SignalFilter의 min_combined_score(0.55)를 통과해야 
        최종적으로 TradeSignal이 생성되고 HistoricalSimulator에 전달됨.
        """
        # ── CASE-1: 둘 다 HOLD ─────────────────────
        if numeric_signal.direction == "HOLD" and pattern_signal.direction == "HOLD": 
            return None 
        
        # ── CASE-2: 방향 불일치 ─────────────────────
        if (
            numeric_signal.direction != "HOLD"
            and pattern_signal.direction != "HOLD"
            and numeric_signal.direction != pattern_signal.direction
        ):
            return None 
        
        # ── CASE-3: 한쪽만 HOLD ─────────────────────
        numeric_combined = pattern_combined = 0.0
        if numeric_signal.direction != "HOLD":
            direction = numeric_signal.direction
            numeric_combined = numeric_signal.confidence * self._numeric_weight
        if pattern_signal.direction != "HOLD":
            direction = pattern_signal.direction
            pattern_combined = pattern_signal.confidence * self._pattern_weight

        # ── CASE-4: 둘다 방향이 동일 ───────────────────
        # 방향은 같으니 아무거나 와도 되고, 
        combined = numeric_combined + pattern_combined
        
        total_latency = numeric_signal.latency_ms + pattern_signal.latency_ms
        
        return TradeSignal(
            ticker=numeric_signal.ticker, 
            timestamp=numeric_signal.timestamp,
            direction=direction,
            combined_score=round(combined, 4),
            num_track_conf=numeric_signal.confidence,
            pat_track_conf=pattern_signal.confidence,
            total_latency_ms=total_latency
        )
            