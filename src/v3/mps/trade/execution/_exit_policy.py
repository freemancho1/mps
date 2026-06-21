""" 
TripleBarrierExitPolicy ─ ExitPolicyPort 구현체 

[반환값]
  - HOLD        : 조건 미충족 → 포지션 유지
  - TAKE_PROFIT : 익절선 도달
  - STOP_LOSS   : 손절선 도달 (브레이크이븐 상향 이후의 스톱 포함)
  - TIME_OUT    : horizon(60) 만료
  - FORCE_CLOSE : 장 마감 전 강제 청산 (당일 청산 원칙)

[장중 도달 판정 ─ 종가가 아닌 고·저가]
  - 종가만 보면 봉 중간 터치 후 되돌아온 청산을 놓쳐 비현실적으로 낙관적.
  - 롱 포지션: high >= 익절선 → TAKE_PROFIT / low <= 손절선 → STOP_LOSS

[동시 도달 시 손절 우선 (보수성)]
  - 한 봉의 (low, high)가 두 장벽을 모두 포함하면 선후를 알 수 없으므로,
    '보수적 비용 모델링' 원칙에 따라 불리한 쪽(손절)을 먼저 가정.
    TripleBarrierLabeler도 동일 규칙(동시 → HOLD 라벨)을 따름.

[브레이크이븐 스톰 ─ 수익성-E]
  - 미실현 수익이 trigger(+1.0%)에 닿으면,
    '다음 봉부터' 손절선을 진입가 X (1 + buffer(=+0.5%))로 상향.
  - buffer(=0.5%) > 왕복비용(=0.41%)이므로 되밀려도 본전 이상으로 마감
    → 풀 손절(-1% - 비용)로 끝나던 거래 일부를 소익·본적으로 구제.
  - '다음 봉부터'인 이유: 같은 봉 안에서 트리거 도달과 
    (상향된 스톱의) 하향 이탈 순서를 분봉으로는 알 수 없음.
    같은 봉 혜택을 주지 않는 것이 보수적
  - 상태는 Order.breakeven_armed / Order.stop_loss에 보관
    (정책 객체 자신은 무상태 → 폴드 간 오염 없음)

[체크 순서]
  1. 만료 (TIME_OUT vs FORCE_CLOSE)
  2. 손절 (보수적으로 익절보다 먼저)
  3. 익절
  4. 청산이 안됐으면 브레이크이븐 트리거 검사 → 다음 봉용 스톱 상향
"""
from __future__ import annotations 

from mps.config import cfg, msg 
from mps.core.types import Bar, Order, ExitHoldReason 
from mps.core.calendar import force_close_datetime 


class TripleBarrierExitPolicy:
    """ ExitPolicyPort 구현체 ─ 고정 TripleBarrier + 브레이크이븐 스톱. """

    def check(self, order: Order, bar: Bar) -> ExitHoldReason:
        # ── 1. 시간 만료
        if bar.timestamp >= order.expire_at:
            force_close_time = force_close_datetime(bar.timestamp.date())
            if bar.timestamp >= force_close_time:
                return cfg.str.force_close
            return cfg.str.time_out
        
        # ── 2. 손절 (동시 도달 대비 익절보다 먼저 검사)
        if bar.low <= order.stop_loss:
            return cfg.str.stop_loss
        
        # ── 3. 익절
        if bar.high >= order.take_profit:
            return cfg.str.take_profit
        
        # ── 4. 브레이크이븐 트리거 - '다음 봉부터' 적용
        if cfg.trade.risk.use_breakeven_stop and not order.breakeven_armed:
            entry = order.price 
            if entry is not None and bar.high >= entry * (1 + cfg.trade.risk.breakeven_trigger):
                new_stop = round(entry * (1 + cfg.trade.risk.breakeven_buffer), 0)
                # max(): 이미 더 높은 스톱이면 낮추지 않음. (단조 상향만 허용)
                order.stop_loss = max(order.stop_loss, new_stop)
                order.breakeven_armed = True 

        return cfg.str.hold 
    
    @staticmethod 
    def exit_reference_price(order: Order, action: str, bar: Bar) -> float:
        """ 
        청산 액션별 '체결 기준가' (실 체결가는 PaperTrader가 슬리피지 반영).

        - TAKE_PROFIT : 시가가 익절선 위에서 출발해도 보수적으로 익절선 가격.
        - STOP_LOSS : 갭 보수화 ─ 시가가 이미 손절선 아래면 손절선이 아닌
                                  '시가' 체결 가정 (스톱 주문은 갭에서 불리하게 체결)
        - TIME_OUT / FORCE_CLOSE : 시장가(현재 봉 종가)
        """
        if action == cfg.str.take_profit:
            return order.take_profit
        if action == cfg.str.stop_loss:
            return min(order.stop_loss, bar.open)
        return bar.close