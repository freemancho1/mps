""" 
SignalAggregator ─ 수치트랙 + 패턴트랙 결합

[결합 원칙: 두 트랙이 동의해야 진입]
  - 단일 트랙만으로 진입하면 오신호(false positive)가 많아짐
  - 두 트랙이 서로 다른 방향을 가르키면 시장 신호가 명확하지 않은 것으로 판단(기권)

[결합 로직]
  1. 둘 다 HOLD → None (신호 없음)
  2. 방향 반대 → None (합의 실패)
  3. 한쪽만 HOLD → 나머지 트랙의 방향 채택, 점수는 한쪽점수로 계산
  4. 동일 방향 → 점수를 두 패턴으로 계산
  5. 점수가 0.55 이상이면 진행

[교체 계획]
  - Phase-2: 단순 가중 평균 → 메타 모델(스태킹) 또는 강화학습 정책으로 발전 예정
  - 두 트랙의 과거 성과를 기반으로 가중치를 동적으로 조정하는 방식 검토
"""
from __future__ import annotations 

from typing import Optional 

from mps.config import cfg, msg 
from mps.core.types import BSDirection
from mps.core.types import NumericSignal, PatternSignal, TradeSignal


class SignalAggregator:
    def __init__(
        self,
        numeric_cw: Optional[float] = None,     # cw: CombinedWeight: 결합 가중치 0.5
        pattern_cw: Optional[float] = None,
    ) -> None:
        self._nw: float = cfg.run.numeric_cw if numeric_cw is None else numeric_cw
        self._pw: float = cfg.run.pattern_cw if pattern_cw is None else pattern_cw

    def combine(self, ns: NumericSignal, ps: PatternSignal) -> Optional[TradeSignal]:
        """ 
        두 트랙의 신호를 결합하여 TradeSignal 또는 None 반환

        combined_score가 min_combined_score(0.55)를 통과해야,
        최종적으로 TradeSignal이 생성되고 HistoricalSimulator에 전달됨.
        """
        # ── CASE-1 둘다 HOLD ─ H, H skip ──────────────
        if ns.direction == cfg.key.HOLD and ps.direction == cfg.key.HOLD: 
            return None
        
        # ── CASE-2 방향 다름 ─ B, S skip ──────────────
        if (
            ns.direction != cfg.key.HOLD and ps.direction != cfg.key.HOLD 
            and ns.direction != ps.direction
        ): 
            return None
        
        # ── CASE-3 한쪽만 HOLD ─ BS, H or BS (하나만 HOLD거나 방향 같음)
        # HOLD가 아닌쪽만 계산됨.
        ns_score = ps_score = 0.0   # Numeric_Signal_Score, Pattern_Signal...
        if ns.direction != cfg.key.HOLD:
            direction = ns.direction 
            ns_score = ns.confidence * self._nw 
        if ps.direction != cfg.key.HOLD:
            direction = ps.direction
            ps_score = ps.confidence * self._pw
        # ── CASE-4 방향이 같음(HOLD 아님) ──────────────
        # direction은 위에서 어느것이 와도 상관없음 (어차피 같음)
        combined_score = ns_score + ps_score 
        total_latency = ns.latency_ms + ps.latency_ms

        return TradeSignal(
            ticker=ns.ticker,
            timestamp=ns.timestamp,
            direction=direction,
            combined_score=combined_score,
            numeric_track_conf=ns.confidence,
            pattern_track_conf=ps.confidence,
            total_latency_ms=total_latency
        )