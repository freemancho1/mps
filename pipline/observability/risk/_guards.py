""" 
리스크 가드 모임 ─ 모델과 완전히 분리된 독립 가드레일.

[설계 원칙]
  - 모델이 얼마나 확신에 차 있더라도 리스크 가드를 우회할 수 없음.
  - 가드는 모델의 출력을 신뢰하지 않음 ─ 언제나 독립적으로 작동.
  
[세 가지 가드]
  - TripleBarrierGuard      : 주문 생성 시 TP/SL/만료 기준가 설정
  - IntradayCloseoutGuard   : 장 마감 강제청산 여부 판단
  - StopLosTakeProfitGuard  : 매 봉마다 보유 포지션의 청산 체크 조건
"""
from __future__ import annotations 

from datetime import datetime, timedelta 
from typing import Literal 

from mps.sys.core.types import Order, TradeSignal
from mps.sys.core.calendar import force_close_dt
from mps.sys import cfg 


class TripleBarrierGuard:
    """ 
    TripleBarrier 기반 주문 생성
    
    [Triple Barrier란]
      - 진입 이후 세 가지 장벽(barrier) 중 하나를 먼저 터치하면 청산:
        · 상단 장벽(+0.5%): 익절
        · 하단 장벽(-0.3%): 손절
        · 시간 장벽(60분): 시간 만료 청산
      - 비대칭 설정(TP=0.5% > SL=0.3%)의 이유:
        손실이 이익보다 더 빨리 차단 → 승률이 낮아도 수익 기대값 유지 가능
      - 만료 시각
        min(진입+60분, 당일 강제청산 시각=15:15)
        → 60분이 지나도 15:15 이전이면 15:15에 강제 청산
    """
    def __init__(self) -> None:
        self._cfg = cfg.triple_barrier
    
    def build_order(
        self,
        signal: TradeSignal,
        entry_price: float, 
        quantity: int, 
        entry_time: datetime,
    ) -> Order:
        """ TradeSignal → Order 변환. TP/SL 절대 가격과 만료 시각을 계산해 Order에 포함."""
        if signal.direction == "BUY":
            # BUY: 상단 +0.5% = 익절, 하단 -0.3% = 손절
            take_profit = entry_price * (1 + self._cfg.take_profit)
            stop_loss = entry_price * (1 - self._cfg.stop_loss)
        else:
            # SELL(공매도): 하단 -0.5% = 익절, 상단 +0.3% = 손절
            take_profit = entry_price * (1 - self._cfg.take_profit)
            stop_loss = entry_price * (1 + self._cfg.stop_loss)

        # 만료 시각: 진입 후 60분 vs 당일 강제청산 중 더 이른 시각
        expire_at = min(
            entry_time + timedelta(minutes=self._cfg.time_horizon),
            force_close_dt(entry_time.date(), cfg.sys.force_close_minutes_before)
        )

        return Order(
            ticker=signal.ticker,
            direction=signal.direction,
            quantity=quantity,
            order_type="MARKET",
            stop_loss=round(stop_loss, 0),      # 원은 정수로 처리
            take_profit=round(take_profit, 0),  # 원화는 정수로 처리
            expire_at=expire_at
        )
    

class IntradayCloseoutGuard:
    """ 
    장 마감 강제청산 
    """