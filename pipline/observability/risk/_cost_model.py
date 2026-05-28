""" 
CostModel ─ 보수적 거래 비용 계산.

[변경 불가 원칙]
  - 비용 모델을 낙관적으로 설정하면 벡테스트에서 수익처럼 보이는 전략이 
    실거래에서 손실이 날 수 있음.
    → 모든 비용 요소를 시장 현실 기준 상한선으로 설정
    
[슬리피지의 책임 분리 ─ 이중계상 방지]
  - 슬리피지는 '가격 효과'이므로 PaperTrader가 체결가에 직접 반영한다
    (BUY는 더 비싸게, SELL은 더 싸게 체결).
  - 따라서 CostModel이 차감하는 '비용'에는 슬리피지를 넣지 않는다.
    여기에 또 넣으면 동일 슬리피지가 (체결가 + 비용) 두 번 빠지는 이중계상이 됨.
  - CostModel이 차감하는 비용 = 실제 현금성 수수료뿐:
      · 매수 시: commission(0.015%)
      · 매도 시: commission(0.015%) + tax(0.18%)
      
[신호 임계 기준치는 별개]
  - min_profitable_return()은 '이 거래가 할 만한가'를 판단하는 기준이므로
    슬리피지까지 포함한 총 왕복 경제비용(≒0.41%)을 반환한다.
    (실제 차감 비용 ≠ 신호 임계 기준치)
"""
from __future__ import annotations 

from mps.sys import cfg 


class CostModel:
    def __init__(self) -> None:
        # 슬리피지는 체결가에 반영되므로 비용율에서는 제외 (이중계상 방지)
        self._buy_rate = cfg.cost.commission_rate
        self._sell_rate = cfg.cost.commission_rate + cfg.cost.tax_rate
    
    def buy_cost(self, price: float, quantity: int) -> float:
        """ 
        매수 시 현금성 비용 = 체결 금액 * commission
        (슬리피지는 PaperTrader 체결가에 이미 반영, 증권거래세는 매수에 없음)
        """
        amount = price * quantity
        return amount * self._buy_rate
    
    def sell_cost(self, price: float, quantity: int) -> float:
        """ 
        매도 시 현금성 비용 = 체결 금액 * (commission + tax)
        (슬리피지는 PaperTrader 체결가에 이미 반영)
        증권거래세(0.18%)는 매도 시에만 부과됨 
        """
        amount = price * quantity 
        return amount * self._sell_rate
    
    def roundtrip_cost(self, price: float, quantity: int) -> float:
        """ 매수 + 매도 왕복 현금성 수수료 총액 (슬리피지 제외) """
        return self.buy_cost(price, quantity) + self.sell_cost(price, quantity)
    
    def min_profitable_return(self) -> float: 
        """ 
        신호가 '할 만한가'를 판단하는 최소 기대수익률 기준.
        실제 차감 비용과 달리, 슬리피지까지 포함한 총 왕복 경제비용(≒0.41%)을 반환.
        → 신호 임계값(combined_score) 설정의 경제적 근거.
        """
        return cfg.cost.roundtrip_cost