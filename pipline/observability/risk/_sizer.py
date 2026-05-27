""" 
PositionSizer ─ 포지션 크기 결정.

[원칙: Kelly Criterion 이전, 고정 비율 먼저]
  - Kelly Criterion(켈리 기준) 
    ─ 장기적으로 자산을 최대화하는 최적 베팅 비율을 계산하는 공식
  - Phase-1에서는 Kelly 기준(최적 베팅 비율 이론)을 사용하지 않음
    → 이유: Kelly는 정확한 승률과 손익비율을 알아야 하는데, 
             Phase-1은 그 통계가 아직 충분히 쌓이지 않았기 때문임
    → 단순히 계죄 대비 10% 고정 비율 적용
    
[계산 방식]
  - max_amount = min(현재 보유 현금, 초기 자본 * 10%)
  - quantity = max_amount // 현재가 (소수점 버림 ─ 주식은 정수 단위)
  
  - 현재 현금이 초기 자본의 10% 이하이면 고게 최대치.
  - 초기 자본 기준 10% 한도 이유: 자본이 늘어나도 과도한 집중 방지
"""
from __future__ import annotations 

from mps.sys import cfg 


class PositionSizer:
    def __init__(
        self,
        capital: float = cfg.run.capital,
        max_position_pct = cfg.sys.max_position_pct
    ) -> None:
        self._capital = capital 
        self._max_pct = max_position_pct
        
    def calc_quantity(self, price: float, available_cash: float) -> int:
        """ 
        매수 가능 수량 계산
        
        available_cash: 현재 미사용 현금 (초기 자본 - 진입 비용 누적)
        price: 현재 봉의 close 가격 (시장가 주문 기준가)
        """
        # 두 한도(현재 금액 vs 초기 자본의 10%) 중 작은 쪽
        max_amount = min(available_cash, self._capital * self._max_pct)
        quantity = int(max_amount // price)
        return max(quantity, 0)
    
    def update_capital(self, capital: float) -> None:
        self._capital = capital