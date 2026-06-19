""" 
RiskManager ─ 진입 승인 게이트 + 사이징 + 주문 생성

[역할]
  - 시뮬레이터는 '신호가 왔으니 승인해 줘'라고 위임만 함.
  - 같은 객체를 향후 실시간 엔진이 그대로 재사용 (백테스트 = 실거래 코드)
  - 거부 사유가 Reject 객체로 구조화되어 '왜 진입하지 않았는가?'도 관측 가능

[승인 절차 ─ approve()]
  1. 진입 컷오프: 체결 시각이 (강제청산 ─ entry_cutoff_minute) 이후면 거부.
     마감 임박 진입은 보유 가능 시간이 짧아 익절 확률은 낮고 왕보비용(0.41%)은 확정
     → 구조적 손실 거래를 사전 차단 [수익성-B]
  2. 일일 손실 한도: 당일 실현 손실이 자본의 -1%를 넘으면 거부 [수익성-F]
  3. 사이징: 신뢰도 비례 수량 계산 [수익성-D] → 0주면 현금 부족으로 거부.
  4. 통과 시 Order 생성 (장벽·만료는 BuildOrder가 체결가 기준으로 산출)
"""
from __future__ import annotations 

from datetime import datetime, timedelta 
from typing import Optional, Union 

from mps.config import cfg, msg 
from mps.core.types import Bar, Order, Reject, TradeSignal 
from mps.core.calendar import force_close_datetime
from mps.trade.execution import PositionSizer, BuildOrder


class RiskManager:
    """ 신호 → 주문 사이의 독립 가드레일 (모델이 무엇이든 손실을 통제). """
    def __init__(
        self,
        capital: Optional[float] = None,
        sizer: Optional[PositionSizer] = None,
        builder: Optional[BuildOrder] = None,
    ) -> None:
        self._capital = cfg.run.init_capital if capital is None else capital 
        self._sizer = PositionSizer(capital=self._capital) if sizer is None else sizer 
        self._builder = BuildOrder() if builder is None else builder 
        
        # 일일 손실 한도 (음수 원화값, 예: -100,000)
        self._daily_loss_limit = -self._capital * cfg.trade.risk.daily_loss_limit_pct

    def can_enter_at(self, work_datetime: datetime, day_pnl: float) -> bool:
        """ 
        해당 시각에 신규 진입이 허용되는지 빠르게 검사.

        시뮬레이터가 '신호 생성 전'에 호출해, 어차피 거부될 시간대에는
        모델 추론 자체를 생략함 (지연시간은 곧 비용)
        """
        return (
            RiskManager.within_entry_window(work_datetime) 
            and day_pnl > self._daily_loss_limit
        )
    
    def approve(
        self,
        signal: TradeSignal,
        fill_bar: Bar,
        fill_price: float,
        available_cash: float,
        day_pnl: float
    ) -> Union[Order, Reject]:
        """ 
        신호를 검증하고 Order 또는 Reject 반환.

        fill_bar / fill_price: 체결이 일어나는 봉(신호 다음 봉)과 체결가.
        day_pnl: 당일 실현 손익 (Portfolio가 추적)
        """
        # 1. 진입 컷오프 ─ 신호 시각이 아닌 '체결 시각' 기준 재검증
        #    (신호 봉과 체결 봉 사이에 컷오프를 넘을 수 있음)
        if not RiskManager.within_entry_window(fill_bar.timestamp):
            return Reject(cfg.str.entry_cutoff, signal)
        
        # 2. 일일 손실 컷오프 ─ 직전 청산으로 한도를 넘었을 수 있어 재검증
        if day_pnl <= self._daily_loss_limit:
            return Reject(cfg.str.daily_loss_limit, signal)
        
        # 3. 신뢰도 비례 사이징
        quantity = self._sizer.calc_quantity(
            fill_price, available_cash, score=signal.combined_score
        )
        if quantity <= 0:
            return Reject(cfg.str.no_cash, signal)
        
        # ── 모두 통과하면 주문 생성
        return self._builder.build(signal, fill_bar, fill_price, quantity)
    

    @staticmethod 
    def within_entry_window(work_datetime: datetime) -> bool:
        """ [수익성-B] 강제청산 entry_cutoff_minutes(30분) 전까지만 진입 허용 """
        cutoff = force_close_datetime(work_datetime.date()) \
            - timedelta(minutes=cfg.trade.risk.entry_cutoff_minutes)
        return work_datetime < cutoff
    

