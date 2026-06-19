""" 
BuildOrder ─ Triple Barrier 기반 주문 생성

[기능]
  1. 장벽(익절·손절) 기준가를 '신호 봉 종가'가 아닌 '실제 체결가(fill_price)'로 산출.
     → 다음 봉 시가 체결 + 슬리피지 반영 후의 가격을 기준으로 ±2%/-1%를 잡아야
        라벨·청산 가드와 가격 기준이 일치함
  2. expire_at 기준 시각을 '신호 봉 시각'이 아닌 '체결 봉 시각'으로 변경
  3. order_id에 날짜 포함: {ticker}_{YYYYMMDD_HHMMSS}
"""
from __future__ import annotations 

from datetime import timedelta 
from typing import Optional 

from mps.config import cfg 
from mps.core.types import Bar, Order, TradeSignal 
from mps.core.calendar import force_close_datetime 


class BuildOrder:
    def __init__(
        self,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        time_horizon: Optional[int] = None,
    ) -> None:
        self._take_profit = cfg.trade.barrier.take_profit \
            if take_profit is None else take_profit 
        self._stop_loss = cfg.trade.barrier.stop_loss \
            if stop_loss is None else stop_loss 
        self._time_horizon = cfg.trade.barrier.time_horizon \
            if time_horizon is None else time_horizon 
        
    def build(
        self,
        signal: TradeSignal, 
        fill_bar: Bar, 
        fill_price: float, 
        quantity: int
    ) -> Order:
        """ 
        TradeSignal → Order 변환

        fill_bar    : 체결이 일어나는 봉 (신호 봉의 '다음' 봉)
        fill_price  : 실제 체결가 (시가 + 슬리피지)
        """
        take_profit = fill_price * (1 + self._take_profit)
        stop_loss = fill_price * (1 - self._stop_loss)

        # 만료 시각: 체결 후 horizon분 vs 당일 강제청산 중 더 이른 시간
        expire_at = min(
            fill_bar.timestamp + timedelta(minutes=self._time_horizon),
            force_close_datetime(fill_bar.timestamp.date(), cfg.market.force_close_minutes),
        )

        return Order(
            ticker=signal.ticker, 
            dir=cfg.str.buy,
            quantity=quantity,
            order_type=cfg.str.market,
            stop_loss=round(stop_loss, 0),
            take_profit=round(take_profit, 0),
            order_id=f"{fill_bar.ticker}_{fill_bar.timestamp.strftime(cfg.sys.datetime_format)}",
            expire_at=expire_at,
            price=fill_price
        )