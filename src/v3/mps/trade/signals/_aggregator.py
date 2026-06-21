""" 
SignalAggregator ─ 수치 + 패턴 트랙 결합 (롱 온리)

[전제: 롱 온리]
  - 두 트랙 모두 BUY(매수 후보) 또는 HOLD(관망)만 출력함. SELL(숏) 신호는 없음.
  - 방향이 '둘 다 관망', '둘 다 매수', '한쪽만 매수' 세 가지로 단순화됨.

[결합 정책]
  - 둘 다 HOLD → 관망(None, 신호없음)
  - require_confluence = True → 두 신호가 모두 BUY일 때만 TradeSignal 생성(기본값)
    require_confluence = False → 한쪽만 BUY 여도 진입 허용 (단일 트랙 모드)

[combined_score 의미 ─ 죽은 경로 제거]
  BUY 트랙들의 '가중 평균 신뢰도'를 활성 가중치 합으로 정규화한 값(0~1).
  - 두 트랙 BUY(가중치 0.5/0.5): (0.5·c_n + 0.5·c_p) / 1.0 = 두 신뢰도 평균
  - 단일 트랙 BUY: 그 트랙의 신뢰도 그대로 (활성 가중치로 나누므로 0.5에 갇히지 않음)

[교체 계획]
  - Phase-2+: 단순 가중 평균 → 메타 모델(스태킹)·강화학습 정책으로 발전 예정.
  - 두 트랙의 과거 성과 기반 동적 가중치(confluence bonus 포함) 검토
"""
from __future__ import annotations

from typing import Optional 

from mps.config import cfg 
from mps.core.types import NumericSignal, PatternSignal, TradeSignal 


class SignalAggregator:
    def __init__(
        self, 
        numeric_cw: Optional[float] = None, # Combined Weight
        pattern_cw: Optional[float] = None, 
        require_confluence: Optional[bool] = None,
    ) -> None:
        self._ncw: float = cfg.trade.signal.numeric_weight \
            if numeric_cw is None else numeric_cw
        self._pcw: float = cfg.trade.signal.pattern_weight \
            if pattern_cw is None else pattern_cw 
        self._require_confluence: bool = cfg.trade.signal.require_confluence \
            if require_confluence is None else require_confluence
        
    def combine(self, ns: NumericSignal, ps: PatternSignal) -> Optional[TradeSignal]:
        """ 
        두 트랙의 신호를 결합하여 TradeSignal 또는 None 반환

        롱 온리이므로 방향이 BUY로 고정되며, combined_score가 
        min_combined_score(=0.55) 이상이어야 최종적으로 SignalFilter를 통과함.
        """
        ns_buy = ns.dir == cfg.str.buy 
        ps_buy = ps.dir == cfg.str.buy 

        # 둘 다 관망 → 신호 없음
        if not ns_buy and not ps_buy:
            return None 
        
        # 합의 요구 모드: 두 트랙이 모두 BUY가 아니면 진입하지 않음.
        if self._require_confluence and not (ns_buy and ps_buy):
            return None 
        
        # BUY 트랙의 가중 평균 계산
        ns_w = self._ncw if ns_buy else 0.0
        ps_w = self._pcw if ps_buy else 0.0
        active_w = ns_w + ps_w 
        combined_score = \
            (ns_w * ns.confidence + ps_w * ps.confidence) / (active_w + cfg.sys.zero)
        total_latency = ns.latency_ms + ps.latency_ms

        return TradeSignal(
            ticker=ns.ticker,
            timestamp=ns.timestamp,
            dir=cfg.str.buy,        # 롱 온리: 진입 방향은 향상 매수
            combined_score=round(combined_score, 5),
            numeric_track_conf=ns.confidence,
            pattern_track_conf=ps.confidence,
            total_latency_ms=total_latency
        )
