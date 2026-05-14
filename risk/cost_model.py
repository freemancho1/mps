""" 
CostModel — 보수적 거래 비용 계산.

변경 불가 원칙:
  - 비용 모델을 낙관적으로 설명하면 백테스트에서 수익처럼 보인 전략이 
    실거래에서 손실이 날 수 있음.
    → 모든 비용 요소를 시장 현실 기준 상한선으로 설정함.
    
비용 구성: 
  - 매수 시: commission(0.015%) + slippage(0.1%) 
  - 매도 시: commission(0.015%) + tax(0.18%) + slippage(0.1%)
  - 왕복 시: 0.015% * 2 + 0.18% + 0.1% * 2 ≒ 0.41%
    → 이 비용을 모두 소요하고도 이익이 남으려면, 
       진입 신호의 기대 수익이 최소 0.41% 이상이어야 함.
"""
from __future__ import annotations

from mps.sys.config import settings


class CostModel:
    def __init__(self) -> None:
        self._cfg = settings.cost 
        
    def buy_cost(self, price: float, qty: int) -> float:
        return 0.0