""" 
PaperTrader ─ 모의투자 실행기 (Phase-1)

[역할]
  - 실제 KIS API 없이 주문 제출 → 체결 결과를 시뮬레이션 함.
  - 슬리피지를 보수적으로 반영하여 실거래와의 괴리를 줄임.
  
[슬리피지 모델]
  - BUY: filled_price = current_price + slippage (불리하게 체결)
  - SELL: filled_price = current_price - slippage (불리하게 체결)
  - slippage = current_price * slipage_rate(0.1%)
  
  실제 시장에서 시장가 주문은 호가창에서 불리한 방향으로 매칭되므로,
  매수는 현재가보다 약간 높게, 매도는 낮게 체결되는 것이 현실적.
  
[한계 ─ Phase-1에서 미구현]
  - 부분 체결(X), 주문 거부(X), 체결 지연(X)
    → (교체 계획) OrderClientPort 인터페이스 구현 → KISOrderClient로 교체 시
"""
from __future__ import annotations 

import uuid 
from datetime import datetime 
from typing import Optional 

from mps.sys.core.types import Order, OrderResult
from mps.sys import cfg


class PaperTrader:
    def __init__(self, slippage_rate: float = cfg.cost.slippage_rate) -> None:
        self._slippage = slippage_rate
        self._results: dict[str, OrderResult] = {}
        
    def submit_order(self, order: Order, current_price: float) -> OrderResult:
        """ 
        주문 제출 후 즉시 체결 결과 반환.
        
        order.order_id가 없으면 UUID로 생성.
        체결가에 슬리피지 반영 (매수: +, 매도: -)
        """
        order_id = order.order_id or str(uuid.uuid4())[:8]
        slippage = current_price * self._slippage
        # 투자자에게 불리한 방향으로 매도/매수 결정
        if order.direction == "BUY":
            filled_price = current_price + slippage # 더 비싸게 매수
        else:
            filled_price = current_price - slippage # 더 싸게 매도
        
        result = OrderResult(
            order_id=order_id,
            status="FILLED",
            filled_price=round(filled_price, 0),    # 원 단위 반올림
            filled_quantity=order.quantity,
            timestamp=datetime.now(),
            slippage=slippage 
        )
        self._results[order_id] = result
        return result 
    
    def cancel_order(self, order_id: str) -> bool:
        """ 주문 취소: Phase-1에서는 즉시 체결이기 때문에 사실상 주문 취소가 사용 안됨 """
        if order_id in self._results:
            self._results[order_id] = OrderResult(
                order_id=order_id,
                status="CANCELLED",
                filled_price=0.0,
                filled_quantity=0,
                timestamp=datetime.now(),
                slippage=0.0
            )
            return True         # 해당 주문이 있어서 정상 주문 취소
        return False            # 해당 주문이 없어서 취소 못함(거래는 이뤄지지 않았음)
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """ 주문 상태 조회 """
        return self._results.get(order_id)