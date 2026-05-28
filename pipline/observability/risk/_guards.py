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
    장 마감 강제청산 ─ 오버나잇 갭 리스크 원천 차단.
    
    KOSPI는 15:30분에 마감이지만 15:15(15분전)분에 강제 청산.
    이유: 시장가 주문 체결에 걸리는 시간 + 마감 직전 유동성 저하 고려
    """
    def should_force_close(self, current_time: datetime) -> bool:
        """ 현재 시각이 강제청산 기준 시간(15:15) 이후면 True """
        target = force_close_dt(
            current_time.date(),
            cfg.sys.force_close_minutes_before
        )
        return current_time >= target
    
    
class StopLossTakeProfitGuard:
    """ 
    포지션 보유 중 매 봉마다 손절·익절·만료 조건 체크
    
    [반환값과 의미]
      - "HOLD"          : 아무 조건도 충족 안 됨 → 포지션 유지
      - "TAKE_PROFIT"   : 익절 기준가 도달 → 수익 실현 청산
      - "STOP_LOSS"     : 손절 기준가 도달 → 손실 한도 청산
      - "TIMEOUT"       : 60분 만료 → 방향없는 청산
      - "FORCE_CLOSE"   : 15:15분 강제 청산 → 오버나잇 방지 청산
      
    [장중(intrabar) 도달 판정 ─ 종가가 아닌 고가/저가 사용]
      - 분봉 '종가'만 보면 봉 중간에 장벽을 터치했다가 되돌아온 청산을 놓쳐
        백테스트가 비현실적으로 낙관적이 된다.
      - 따라서 봉의 high/low로 장벽 도달을 판정한다.
        · BUY  포지션: high >= 익절선 → TAKE_PROFIT, low <= 손절선 → STOP_LOSS
        · SELL 포지션: low <= 익절선 → TAKE_PROFIT, high >= 손절선 → STOP_LOSS

    [동시 도달 시 손절 우선 (보수적 가정)]
      - 한 봉의 (low, high) 범위가 익절선과 손절선을 동시에 포함하면
        어느 쪽이 먼저 닿았는지 분봉만으로는 알 수 없다.
      - '보수적 비용 모델링' 원칙에 따라 불리한 쪽(손절)이 먼저 닿았다고 가정한다.
      
    [체크 순서]
      1. 만료 시각 초과 여부 (TIMEOUT vs FORCE_CLOSE)
      2. 손절 도달 (STOP_LOSS) ─ 보수적으로 먼저 검사
      3. 익절 도달 (TAKE_PROFIT)
    """
    def check(
        self, order: Order, high: float, low: float, current_time: datetime,
    ) -> Literal["HOLD", "STOP_LOSS", "TAKE_PROFIT", "TIMEOUT", "FORCE_CLOSE"]:
        # ── 1. 만료 체크 ───────────────────────
        if current_time >= order.expire_at:
            # 15:15 이후면 FORCE_CLOSE, 60분 만료이면 TIMEOUT
            if IntradayCloseoutGuard().should_force_close(current_time):
                return "FORCE_CLOSE"
            return "TIMEOUT" 
        
        # ── 2. BUY 포지션 (손절 우선 → 익절) ───────────────────────
        if order.direction == "BUY":
            if low <= order.stop_loss:          # 저가가 손절선 이하로 내려감
                return "STOP_LOSS"
            if high >= order.take_profit:        # 고가가 익절선 이상으로 올라감
                return "TAKE_PROFIT"
            
        # ── 3. SELL 포지션 (손절 우선 → 익절) ───────────────────────
        else:
            if high >= order.stop_loss:          # 고가 상승이 공매도 손실
                return "STOP_LOSS"
            if low <= order.take_profit:         # 저가 하락이 공매도 이익
                return "TAKE_PROFIT" 
            
        return "HOLD"