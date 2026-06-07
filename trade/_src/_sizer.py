""" 
PositionSizer ─ 포지션 크기 결정.

[원칙: Kelly Criterion 이전, 고정 비율 먼저]
  - Kelly Criterion (켈리 기준)
    ─ 장기적으로 자산을 최대화하는 최적 배팅 비율을 계산하는 공식
  - Phase-1에서는 Kelly 기준(최적 배팅 비율 이론)을 사용하지 않음
    · 이유: Kelly는 정확한 승률과 손익비율을 알아야 하는데,
            Phase-1은 그 통계가 아직 충분히 쌓이지 않았기 때문임
    · Phase-1에서는 단순히 초기 투자금의 10% 고정 비율 적용

[계산 방식]
  - max_amount = min(보유 현금, 초기 자본 * 10%)
    → 초기 자본 10%: 거래를 통해 보유 현금이 늘어나도 과도한 집중 투자 방지
  - quantity = max_amount // 현재가 (소숫점 버림)
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
        self._capital = cfg.run.init_capital \
            if capital is None else capital
        self._max_pct = cfg.run.max_capital_pct \
            if max_capital_pct is None else max_capital_pct
        
    def calc_quantity(self, price: float, available_cash: float) -> int:
        """ 매수 가능 수량 계산 """
        max_amount = min(available_cash, self._capital * self._max_pct)
        quantity = int(max_amount // price)
        return max(quantity, 0)
    
    def set_capital(self, capital: float) -> None:
        self._capital = capital

    def set_max_capital_pct(self, max_capital_pct: float) -> None:
        self._max_pct = max_capital_pct