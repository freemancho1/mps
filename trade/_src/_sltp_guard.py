""" 
포지션 보유 중 매 봉마다 손절·익절·만료 조건 체크

[반환값의 의미]
  - "HOLD"          : 아무 조건도 충족 안 됨 → 포지션 유지
  - "TAKE_PROFIT"   : 익절 기준가 도달 → 수익 실현 청산
  - "STOP_LOSS"     : 손절 기준가 도달 → 손실 한도 청산
  - "TIMEOUT"       : 60분(만료 최대치=60분) → 방향성 없이 현재가 청산
  - "FORCE_CLOSE"   : 15:15분 강제 청산 → 오버나잇 방지 청산

[장중 도달 판정 ─ 종가가 아닌 고·저가 사용]
  - 분봉 '종가'만 보면 봉 중간에 장벽을 터치했다가 되돌아온 청산을 놓쳐,
    테스트가 비현실적으로 낙관적으로 됨.
  - 따라서, 봉의 high·low로 장벽 도달을 판정해야 함.
    · BUY 포지션: high >= 익절선 → TAKE_PROFIT, low <= 손절선 → STOP_LOSS
    · SELL 포지션: low <= 익절선 → TAKE_PROFIT, high >= 손절선 → STOP_LOSS

[동시 도달 시 손절 우선 (보수적 가정)]
  - 한 봉의 (low, high) 범위가 익절선과 손절선을 동시에 포함하면,
    어느 쪽이 먼저 닿았는지 분봉만으로 알 수 없기 때문에,
    "보수적 비용 모델링" 원칙에 따라 불리한 쪽(손절)이 먼저 닿았다고 가정

[체크 순서]
  1. 만료 시각 초과 여부 (TIMEOUT vs FORCE_CLOSE)
  2. 손절 도달 (STOP_LOSS) ─ 보수적으로 먼저 검사
  3. 익절 도달 (TAKE_PROFIT)
"""
from __future__ import annotations 

from datetime import datetime 

from mps.core.types import Bar, Order, ExitHoldReason
from mps.core.calendar import force_close_dt
from mps.config import cfg


class StopLossTakeProfitGuard:
    def check(self, order: Order, bar: Bar) -> ExitHoldReason:
        # ── 1. 시간 체크 ────────────
        if bar.timestamp >= order.expire_at:
            force_close_time = force_close_dt(
                bar.timestamp.date(),
                cfg.run.force_close_minutes
            )
            if bar.timestamp >= force_close_time:
                return cfg.str.force_close
            return cfg.str.time_out
        
        # ── 2. BUY 포지션 ───────────
        if order.direction == cfg.key.BUY:
            if bar.low <= order.stop_loss:      # 저가가 손절선 이하로 내려감
                return cfg.str.stop_loss
            if bar.high >= order.take_profit:   # 고가가 익절선 이상으로 올라감
                return cfg.str.take_profit
            
        # ── 3. SELL 포지션 ───────────
        else:
            if bar.high >= order.stop_loss:     # 고가 상승이 공매도 손실
                return cfg.str.stop_loss
            if bar.low <= order.take_profit:    # 저가 하락이 공매도 이익
                return cfg.str.take_profit
            
        return cfg.str.hold