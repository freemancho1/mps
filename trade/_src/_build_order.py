""" 
TripleBarrier 기반 주문 생성

[TripleBarrier is]
  - 진입 후 세가지 장벽(barrier) 중 하나를 먼저 터치하면 청산(익·손절)
    ─ 상단 장벽(+2%, 익절), 하단 장벽(-1%, 손절), 시간 장벽(60분, 시간만료 청산)
  - 비대칭(이익 +2%, 손실 -1%) 설정 이유:
    ─ 손실이 이익보다 더 빨리 차단 → 승률이 낮아도 수익 기대값 유지 위해 필요
    ─ Phase-3에서는 이익에 대해서 한도를 정할 필요가 있을까 싶은데,
       이익 한도 없에고, 이익일 경우 시간 제한 없에는 부분도 검토 필요
  - 만료 시각
    ─ min(진입+60분, 당일 강제청산 시각(=15:15))
       → 60분이 지나 15:15 이후면 15:15에 강제 청산
"""
from __future__ import annotations 

from datetime import datetime, timedelta 
from typing import Optional, cast

from mps.config import cfg 
from mps.core.types import Bar, Order, TradeSignal, BSDirection
from mps.core.calendar import force_close_dt


class BuildOrder:
    def __init__(
        self,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None, 
        time_horizon: Optional[int] = None,
    ) -> None:
        self._take_profit = cfg.run.take_profit if take_profit is None else take_profit
        self._stop_loss = cfg.run.stop_loss if stop_loss is None else stop_loss 
        self._time_horizon = cfg.run.time_horizon if time_horizon is None else time_horizon

    def build(self, signal: TradeSignal, bar: Bar, quantity: int) -> Order:
        """ 
        TradeSignal → Order 변환 
        - TakeProfit, StopLoss 및 만료 시각을 계산해 Order에 포함
        """
        if signal.direction == cfg.key.BUY:
            # BUY: 상단 +2% = 익절, 하단 -1% = 손절
            take_profit = bar.close * (1 + self._take_profit)
            stop_loss = bar.close * (1 - self._stop_loss)
        else:
            # SELL(공매도): 하단 -2% = 익절, 상단 +1% = 손절
            take_profit = bar.close * (1 - self._take_profit)
            stop_loss = bar.close * (1 + self._stop_loss)

        # 만료 시각: 진입 후 60분 vs 당일 강제 청산 중 더 이른 시각
        expire_at = min(
            bar.timestamp + timedelta(minutes=self._time_horizon),
            force_close_dt(bar.timestamp.date(), cfg.run.force_close_minutes)
        )

        return Order(
            ticker=signal.ticker,
            direction=cast(BSDirection, signal.direction),  # Direction → BSDirection 변환
            quantity=quantity,
            order_type=cfg.str.market,
            stop_loss=round(stop_loss, 0),                  # 원화는 정수형이니 반올림
            take_profit=round(take_profit, 0),
            order_id=f"{bar.ticker}_{bar.timestamp.strftime('%H%M%S')}",
            expire_at=expire_at
        )