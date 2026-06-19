""" 
PaperTrader ─ 모의투자 실행기 (KIS API 없이 가상 체결)

[슬리피지 모델]
  - BUY : filled_price = ref_price + slippage (불리하게)
  - SELL: filled_price = ref_price - slippage (불리하게)
  - slippage = ref_price * slippage_rate(=0.1%)

[한계] 부분 체결, 주문 거부, 체결 지연 → KISOrderClient 교체 시 구현
"""
from __future__ import annotations 

import uuid 
from datetime import datetime 
from typing import Optional 

from mps.config import cfg 
from mps.core.types import Order, OrderResult 


class PaperTrader:
    def __init__(self, slippage_rate: Optional[float] = None) -> None:
        self._slippage_rate = cfg.trade.cost.slippage_rate \
            if slippage_rate is None else slippage_rate
        self._results: dict[str, OrderResult] = {}

    def submit_order(
        self, 
        order: Order,
        ref_price: float, 
        curr_datetime: Optional[datetime] = None,
    ) -> OrderResult:
        """ 
        주문 체결 후 즉시 체결 결과 반환.

        ref_price: 체결 기준가 (진입: 다음 봉 시가 / 청산: 장벽가 또는 봉 종가)
        curr_datetime: 시뮬레이션 시각 (None이면 실거래로 간주하고 현재 시각 사용)
        """
        slippage = ref_price * self._slippage_rate 
        # 투자자에게 불리한 방향으로 체결
        filled_price = ref_price + (slippage if order.dir == cfg.str.buy else -slippage)

        result = OrderResult(
            order_id=order.order_id or str(uuid.uuid4()),
            status=cfg.str.filled,
            filled_price=round(filled_price, 0),
            filled_quantity=order.quantity,
            timestamp=datetime.now() if curr_datetime is None else curr_datetime,
            slippage=slippage,
        )
        self._results[result.order_id] = result 
        return result 
    
    def cancel_order(
        self, 
        order_id: str, 
        curr_datetime: Optional[datetime] = None,
    ) -> bool:
        """ 주문 취소 ─ 현재는 즉시 체결 거래라 실사용은 없음 (인터페이스 유지용) """
        if order_id in self._results:
            self._results[order_id] = OrderResult(
                order_id=order_id,
                status=cfg.str.cancelled,
                filled_price=0.0,
                filled_quantity=0,
                timestamp=datetime.now() if curr_datetime is None else curr_datetime,
                slippage=0.0
            )
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        return self._results.get(order_id)