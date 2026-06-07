""" 
PaperTrader ─ 모의투자 실행기 (KIS API 연계 없이 가상 거래 실행)

[역할]
  - 실제 KIS API 없이 모의 주문 체결 ─ 체결 결과를 시뮬레이션 함.
  - 슬리피지를 보수적으로 반영하여 실거래와의 괴리를 줄임.

[슬리피지 모델]
  - BUY: filled_price = current_price + slippage (불리하게 체결)
  - SELL: filled_price = current_price - slippage (불리하게 체결)
  - slippage = current_price * slippage_rate(0.1%)

  실제 시장에서 시장가 주문은 호가창에서 불리한 방향으로 매칭되므로,
  매수는 현재가보다 약간 높게, 매도는 낮게 체결되는 것이 현실적임.

[한계 ─ Phase-1에서 미구현]
  - 부분 체결(X), 주문 거부(X), 체결 지연(X)
    → (교체 계획) OrderClientPort 인터페이스 구현 또는 KISOrderClient 교체 시
"""
from __future__ import annotations 

import uuid 
from datetime import datetime 
from typing import Optional 

from mps.config import cfg 
from mps.core.types import Order, OrderResult


class PaperTrader:
    def __init__(self, slippage_rate: Optional[float] = None) -> None:
        self._slippage_rate = \
            cfg.run.slippage_rate if slippage_rate is None else slippage_rate
        self._results: dict[str, OrderResult] = {}

    def submit_order(self, order: Order, curr_price: float) -> OrderResult:
        """ 
        주문 제출 후 즉시 체결 결과 반환

        order.order_id가 없으면 uuid로 생성. (BuildOrder에서 생성함)
        체결가에 슬리피지 반영 (매수에는 슬리피지를 더하고 매도에는 뺌 → 슬리피지를 비용 취급)
        """
        slippage = curr_price * self._slippage_rate
        # 투자자에게 불리한 방향으로 매도·매수 비용 결정
        filled_price = curr_price + \
            (slippage if order.direction == cfg.key.BUY else -slippage)
        
        result = OrderResult(
            order_id=order.order_id,
            status=cfg.str.filled,
            filled_price=round(filled_price, 0),    # 원단위 반올림
            filled_quantity=order.quantity,
            timestamp=datetime.now(),
            slippage=slippage 
        )
        self._results[order.order_id] = result 
        return result 
    
    def cancel_order(self, order_id: str) -> bool:
        """ 주문 취소: Phase-1에서는 즉시 체결이기 때문에 이 코드가 필요없음 """
        if order_id in self._results:
            self._results[order_id] = OrderResult(
                order_id=order_id,
                status=cfg.str.cancelled,
                filled_price=0.0,
                filled_quantity=0,
                timestamp=datetime.now(),
                slippage=0.0
            )
            return True 
        return False    # 취소 요청 주문이 없어서 취소 못함(거래는 없었음)
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        return self._results.get(order_id)