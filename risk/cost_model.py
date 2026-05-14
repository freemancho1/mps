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
        """ 
        매수 시 비용 = 체결금액 * (수수료 + 슬리피지)

        매수에는 증권거래세가 없으므로 commission + slippage 만 포함
        """
        amount = price * qty 
        print(
            f"  - 거래금액: {amount:,}, "
            f"수수료: {self._cfg.commission_rate}, "
            f"슬리피지 수수료: {self._cfg.slippage_rate}"
        )
        return amount * (
            self._cfg.commission_rate + 
            self._cfg.slippage_rate
        )
    
    def sell_cost(self, price: float, qty: int) -> float:
        """ 
        매도 시 비용 = 체결금액 * (수수료 + 증권거래세 + 슬리피지)

        증권거래세는 매도 시에만 부과됨 (코스피 2024년 기준)
        """
        amount = price * qty 
        return amount * (
            self._cfg.commission_rate +
            self._cfg.tax_rate +
            self._cfg.slippage_rate
        )
    
    def roundtrip_cost(self, price: float, qty: int) -> float:
        """ 매수 + 매도 왕복 총 비용. """
        return self.buy_cost(price, qty) + self.sell_cost(price, qty)
    
    def min_profitable_return(self) -> float:
        """ 
        이 수익률 이상이여야 왕복 비용을 넘어 실제 이익이 발생함
        - 신호 임계값 combined_score >= 0.55 설정의 경제적 근거
        """
        return self._cfg.roundtrip_cost



