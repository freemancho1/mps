""" 신호 관련 필터와 지연시간 관리자 구현 """
from __future__ import annotations

from typing import Optional 

from mps.config import cfg 
from mps.core.types import TradeSignal


""" 
LatencyGuard ─ 합산 지연시간 초과 신호 폐기

[원칙: "지연시간은 곧 비용"]
  - 분봉 단타에서 신호 생성부터 체결까지 5초 이상 걸리면,
    봉의 정보가 이미 시장에 반영되어 효과가 사라짐.
    → 지연된(지나간) 신호에는 반응하지 않는게 좋음.

[임계값] 
  - max_total_ms = max_inference(3000) + max_network(1000) + max_order(1000) = 5000ms
  - Phase-1(백테스트)에서는 모든 추론이 수ms 이내이므로 모두 통과
    → 실거래 전환 시 실제 네트워크 지연을 측정하여 임계값을 보정해야 함.
"""
class LatencyFilter:
    def __init__(self, max_latency_ms: Optional[float] = None) -> None:
        self._max_latency_ms = max_latency_ms or cfg.run.max_latency_ms

    def _allow(self, signal: TradeSignal) -> bool:
        return signal.total_latency_ms <= self._max_latency_ms
    
    def filter(self, signal: Optional[TradeSignal] = None) -> Optional[TradeSignal]:
        """ 
        None이 들어오거나 지연시간 초과면 None을 반환

        HistoricalSimulator에서는 None 체크가 연쇄적으로 이뤄지므로 중간에
        None이 발생하면 해당 봉의 신호 처리가 자동으로 중단됨.
        """
        if signal is None:
            return None 
        return signal if self._allow(signal) else None 

""" 
SignalFilter ─ 합의 점수 임계값 미만의 약한 신호 제거

[임계값 0.55의 의미]
  - combined_score = numeric_confidence * 0.5 + pattern_confidence * 0.5
    → 0.55를 달성하려면 두 트랙 평균 신뢰도가 55% 이상이어야 함.
    → 왕복비용(최소 0.41%)을 커버하려면 최소 이 신뢰도가 필요하다고 가정

[LatencyGuard와의 우선순위]
  - SignalAggregator → LatencyFilter → SingalFilter 순으로 적용
    → 지연이 과도한 신호를 먼저 제거하고, 그 다음 약한 신호를 제거
"""
class SignalFilter:
    def __init__(self, min_score: Optional[float] = None) -> None:
        self._min_score = cfg.run.min_combined_score if min_score is None else min_score

    def _allow(self, signal: TradeSignal) -> bool:
        return signal.combined_score >= self._min_score
    
    def filter(self, signal: Optional[TradeSignal] = None) -> Optional[TradeSignal]:
        if signal is None:
            return None 
        # 임계값 미만 신호 → None 반환으로 파이프라인 종료
        return signal if self._allow(signal) else None