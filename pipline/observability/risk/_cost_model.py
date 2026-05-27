""" 
CostModel ─ 보수적 거래 비용 계산.

[변경 불가 원칙]
  - 비용 모델을 낙관적으로 설정하면 벡테스트에서 수익처럼 보이는 전략이 
    실거래에서 손실이 날 수 있음.
    → 모든 비용 요소를 시장 현실 기준 상한선으로 설정
    
[비용 구성]
  - 매수 시: commission(0.015%) + slippage(0.1%)
  - 매도 시: commission(0.015%) + slippage(0.1%) + tax(0.18%)
  - 왕복 시: 2 x (0.015% + 0.1%) + 0.18% = 0.41%
  
  → 이 비용을 모두 회수하고도 이익이 남으려면 진입 신호의 기대 수익이 최소
     0.41% 이상이어야 함.
"""
from __future__ import annotations 

from mps.sys import cfg 


class CostModel:
    def __init__(self) -> None:
        self._buy_rate = cfg.cost.commission_rate + cfg.cost.slippage_rate
        self._sell_rate = self._buy_rate + cfg.cost.tax_rate
    
    def buy_cost(self, price: float, quantity: int) -> float:
        """ 
        매수 시 비용 = 체결 금액 * (수수료 + 슬리피지)
        
        매수에는 증권거래세가 없으므로 commission + slippage만 포함.
        """
        amount = price * quantity
        return amount * self._buy_rate
    
    def sell_cost(self, price: float, quantity: int) -> float:
        """ 
        매도 시 비용 = 체결 금액 * (수수료 + 슬리피지 + 세금)
        
        증권거래세(0.18%)는 매도 시에만 부과됨 
        """
        amount = price * quantity 
        return amount * self._sell_rate
    
    def roundtrip_cost(self, price: float, quantity: int) -> float:
        """ 매수 + 매도 왕복 총 비용 """
        return self.buy_cost(price, quantity) + self.sell_cost(price, quantity)
    
    def min_profitable_return(self) -> float: 
        """ 
        이 수익률 이상이어야 왕복 비용을 넘어 실제 이익이 발생하는 금액.
        → 신호 임계값 combined_score >= 0.55 설정의 경제적 근거.
        """
        return cfg.cost.roundtrip_cost