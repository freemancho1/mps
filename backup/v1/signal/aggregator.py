""" 
SignalAggregator: 수치트랙 + 패턴트랙 신호 결합

[결합 원칙: "두 트랙이 동의해야 진입"]
- 단일 트랙만으로 진입하면 오신호(false positive)가 많아짐.
- 두 트랙이 서로 다른 방향을 가르키면 시장 신호가 명확하지 않은 것으로 판단하여 기권

[결합 로직]
- 둘 다 HOLD → None (신호 없음)
- 방향이 다름(반대) → None (합의 실패)
- 한쪽만 HOLD → 나머지 트랙의 방향 선택, 점수는 해당 트랙 점수 * 가중치(0.5)
- 방향이 동일 → combined = num_conf * 0.5 + pat_conf * 0.5

[교체 계획]
- Phase 2: 단순 가중 평균 → 메타 모델(스태킹) 또는 강화학습 정책으로 발전 예정
- 두 트랙의 과거 성과를 기반으로 가중치를 동적 조정하는 방식도 검토 중
"""
from __future__ import annotations

from typing import cast 

from mps.data.types import BSDirection, NumericalSignal, PatternSignal, TradeSignal

# 방향 → 수치 매핑 (현재 코드에서도 직접 사용하지 않으나 향후 방향 비교 일반화 시 활용)
_DIR_SCORE = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}


class SignalAggregator:
    def __init__(self, num_weight: float = 0.5, pat_weight: float = 0.5) -> None:
        self._num_weight = num_weight
        self._pat_weight = pat_weight
        
    def combine(self, num: NumericalSignal, pat: PatternSignal) -> TradeSignal | None:
        """ 
        두 트랙 신호를 결합하여 TradeSignal 또는 None 반환.
        
        반환이 None인 경우:
        - 둘 다 HOLD: 아무 신호도 없음.
        - 방향 충돌: 시장 방향 불명확 → 기권이 최선(모르겠으면 기권)
        
        combined_score가 SignalFilter의 min_combined_score(0.55)를 통과해야 
        최종 TradeSignal이 HistoricalSimulator에 전달됨.
        """
        
        # --- Case 1: 둘 다 HOLD ----------------------------------------------
        if num.direction == "HOLD" and pat.direction == "HOLD":
            return None 
        
        # --- Case 2: 방향 충돌 ------------------------------------------------
        if (
            num.direction != "HOLD" 
            and pat.direction != "HOLD"
            and num.direction != pat.direction
        ):
            return None 
        
        # --- Case 3: 한쪽만 HOLD ----------------------------------------------
        if num.direction == "HOLD":
            # 패턴 트랙만 방향 있음, 점수 = pat_conf * self._pw 
            direction = pat.direction
            combined = pat.confidence * self._pat_weight
        elif pat.direction == "HOLD":
            # 수치 트랙만 방향 있음, 점수 = num_conf * self._nw
            direction = num.direction
            combined = num.confidence * self._num_weight
        else:
            # 두 트랙 모두 동일 방향
            direction = pat.direction
            combined = num.confidence * self._num_weight + pat.confidence * self._pat_weight
        
        total_latency = num.latency_ms + pat.latency_ms
        
        return TradeSignal(
            ticker=num.ticker, 
            timestamp=num.timestamp, 
            direction=cast(BSDirection, direction), 
            combined_score=round(combined, 4),
            num_track_conf=num.confidence,
            pat_track_conf=pat.confidence,
            total_latency_ms=total_latency
        )
        