""" 
PositionSizer ─ 포지션 크기 결정

[기본 원칙: Kelly 이전, 고정 비율 먼저]
  - Kelly Criterion은 정확한 승률·손익비 추정치가 필요한데, 
    그 통계는 실 데이터 기반 운영이 쌓인 후에야 신뢰할 수 있음.
  - 기본은 '초기 자본의 10%' 고정 비율:
    max_amount = min(보유 현금, 초기자본 * 10% * 확신 배율)
    quantity = max_amount // 현재가 ─ 내림
    → 초기자본이 기준이 되는 이유는 수익이 현금으로 불어도 1회 베팅이 따라서 커지는
       복리 폭주(과집중)를 막기 위함

[수익성-D: 신뢰도 비례 사이징 (Conviction Sizing)]
  - 임계값을 간신히 넘은 신호와 score가 1에 가까운 신호에 같은 금액을 거는 것은
    기대값 관점에서 비효율
  - score를 [min_combined_score, 1.0] 구간에서 선형 보간해 투입 배율을
    [conviction_min_factor(0.7) ~ conviction_max_factor(1.3)]로 조정
    → factor = min_f + (max_f - min_f) * (score - 임계값) / (1 - 임계값)
  - Kelly의 '확신 비례 베팅' 정신의 보수적 단순화 (중간 단계).
"""
from __future__ import annotations 

from typing import Optional 

from mps.config import cfg


class PositionSizer:
    def __init__(
        self,
        capital: Optional[float] = None, 
        max_capital_pct: Optional[float] = None,
    ) -> None:
        self._capital = capital or cfg.run.init_capital         # 0인 경우 초기값
        self._max_pct = cfg.trade.risk.max_capital_pct \
            if max_capital_pct is None else max_capital_pct     # 0인 경우 0 사용
        
    def calc_quantity(
        self,
        price: float,
        available_cash: float, 
        score: Optional[float] = None,
    ) -> int:
        """ 
        매수 가능 수량 계산.

        score: TradeSignal.combined_score.
               None이거나 conviction_sizing=False이면 고정 비율 그대로 사용.
        """
        base_amount = self._capital * self._max_pct

        if score is not None and cfg.trade.risk.conviction_sizing:
            base_amount *= PositionSizer._conviction_factor(score)

        max_amount = min(available_cash, base_amount)
        quantity = int(max_amount // price)
        return max(quantity, 0)


    def set_capital(self, capital: float) -> None:
        self._capital = capital 

    def set_max_capital_pct(self, max_capital_pct: float) -> None:
        self._max_pct = max_capital_pct


    @staticmethod 
    def _conviction_factor(score: float) -> float:
        """ score → 투입 비율 [min_factor, max_factor] 선형 매핑 """
        min_score = cfg.trade.min_combined_score

        if score <= min_score:
            return cfg.trade.risk.conviction_min_factor
        
        ratio = min(1.0, (score - min_score) / max(1.0 - min_score, cfg.sys.zero))
        add_factor = (cfg.trade.risk.conviction_max_factor - cfg.trade.risk.conviction_min_factor) * ratio
        
        return cfg.trade.risk.conviction_min_factor + add_factor