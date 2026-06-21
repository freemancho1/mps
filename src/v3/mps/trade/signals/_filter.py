""" 
신호 필터 ─ LatencyFilter(지연 가드) + ScoreFilter(점수 가드)

[적용 순서]
  SignalAggregator → LatencyFilter → ScoreFilter
  (지연과다 신호 먼저 제거하고 점수 낮은 신호 제거)
"""
from __future__ import annotations

from typing import Optional 

from mps.config import cfg 
from mps.core.types import TradeSignal 


class LatencyFilter:
    """ 
    합산 지연시간 초과 신호 폐기 
    ("지연시간은 곧 비용" 원칙과 함께 시간이 지난 신호는 정확도가 떨어짐)

    [임계값] max_latency_ms = 5000ms
      - 백테스트에서는 추론이 수 ms라 모두 통과. (통신 지연등이 없음)
      - 실거래 전환 시 실제 네트워크·주문 지연을 측정해 보정 필요
    """
    def __init__(self, max_latency_ms: Optional[float] = None) -> None:
        self._max_latency_ms = cfg.trade.signal.max_latency_ms \
            if max_latency_ms is None else max_latency_ms
        
    def _allow(self, signal: TradeSignal) -> bool:
        return signal.total_latency_ms <= self._max_latency_ms
    
    def filter(self, signal: Optional[TradeSignal] = None) -> Optional[TradeSignal]:
        """ None 입력 또는 지연 초과 → None (파이프라인 체인 중단) """
        if signal is None: 
            return None 
        return signal if self._allow(signal) else None


class ScoreFilter:
    """ 합의 점수가 임계값 미만인 약한 신호 제거 (과거래 방지). """
    def __init__(self, min_score: Optional[float] = None) -> None:
        self._min_score = cfg.trade.min_combined_score \
            if min_score is None else min_score 
        
    def _allow(self, signal: TradeSignal) -> bool:
        return signal.combined_score >= self._min_score
    
    def filter(self, signal: Optional[TradeSignal] = None) -> Optional[TradeSignal]:
        if signal is None:
            return None 
        return signal if self._allow(signal) else None 