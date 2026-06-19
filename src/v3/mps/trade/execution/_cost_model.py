""" 
CostModel ─ 보수적으로 왕복 거래에 소요된 비용을 계산

[변경 불가 원칙]
  - 비용 모델을 낙관적으로 설정하면, 벡테스트에서 수익처럼 보이는 전략이,
    실 거래에서 손실이 날 수 있음.
  - 모든 비용 요소를 시장 현실 기준 상한선으로 설정 (모의 투자에 한함)

[슬리피지 책임 분리 ─ 이중계상 방지]
  - 슬리피지는 '가격 효과'이므로 PaperTrader에서 체결가에 이미 반영함.
    ─ BUY는 더 비싸게(+), SELL은 더 싸게(-)로 체결
  - 따라서 CostModel이 차감하는 '비용'에는 슬리피지를 넣지 않음 (이중계상 방지).
  - CostModel이 차감하는 비용 = 실제 현금성 수수료:
    · 매수 시: 거래소 수수료 (0.015%)
    · 매도 시: 거래소 수수료 (0.015%) + 세금(0.18%)

[신호 임계 기준치는 별개]
  - min_profitable_return()은 '이 거래가 할만한가?'를 판단하는 기준이므로,
    슬리피지까지 포함한 왕복 경제비용에 대한 비율(약 0.41%)을 리턴함.
"""
from __future__ import annotations 

from mps.config import cfg 


class CostModel:
    def __init__(self) -> None:
        self._buy_rate = cfg.trade.cost.commission_rate
        self._sell_rate = cfg.trade.cost.commission_rate + cfg.trade.cost.tax_rate
        # 왕복비용에는 슬리피지 비율이 포함되어 있음.
        self._roundtrip_rate_with_slippage = cfg.trade.cost.roundtrip_rate

    def buy_cost(self, price: float, quantity: int) -> float:
        amount = price * quantity 
        return amount * self._buy_rate 
    
    def sell_cost(self, price: float, quantity: int) -> float:
        amount = price * quantity 
        return amount * self._sell_rate
    
    def roundtrip_cost(self, price: float, quantity: int) -> float:
        return self.buy_cost(price, quantity) + self.sell_cost(price, quantity)
    
    def min_profitable_return(self) -> float:
        """ 
        이 신호가 '할 만 한가?'를 판단하는 최소 기대수익률을 확인하는 기준으로,
        PaperTrader와 별도로 동작하기 때문에 슬리피지를 포함한 비율을 전달해야 함.
        ─ 신호 임계값(combined_score) 설정의 경제적 근거
        """
        return self._roundtrip_rate_with_slippage