""" 
Portfolio ─ 현금·포지션·거래 기록·일일 손익 원장

'돈과 포지션의 장보'라는 단일 책임만 가지며 일일 손익 추적을 캡슐화.
─ 시뮬레이터와 (향후) 실시간 엔진이 동일 원장 코드를 공유

[현금 흐름 규칙]
  - 진입: cash -= 체결가 * 수량 + 매수 수수료
  - 청산: cash += 진입원금 + 총 손익(gross) - 매도 수수료
    (매수 수수료는 진입 때 이미 차감되므로 청산 때 다시 빼지 않음.)
  - 슬리피지는 체결가에 반영된 가격 효과이므로 원장에 별도 항목이 없음.
"""
from __future__ import annotations 

from datetime import date, datetime 
from typing import Optional 

from mps.config import cfg, msg 
from mps.core.types import Order, TradeRecord 


class Portfolio:
    
    def __init__(self, capital: Optional[float] = None) -> None:
        self._init_capital = capital or cfg.run.init_capital
        self._cash: float = self._init_capital
        
        # 단일 포지션 원칙: 보유 주문은 0 또는 1개
        self._open_order: Optional[Order] = None 
        self._entry_time: Optional[datetime] = None 
        
        # 매수+매도가 완성된 완결 거래 누적 정보
        self._trades: list[TradeRecord] = []
        
        # 일일 실현 손익 (일일 손실 한도가 근거)
        self._curr_date: Optional[date] = None 
        self._day_pnl: float = 0.0
        
    # ── 조회 함수:
    @property 
    def init_capital(self) -> float:
        return self._init_capital
    
    @property 
    def cash(self) -> float:
        return self._cash 
    
    @property 
    def has_position(self) -> bool:
        return self._open_order is not None
    
    @property 
    def open_order(self) -> Optional[Order]:
        return self._open_order
    
    @property 
    def entry_time(self) -> Optional[datetime]:
        return self._entry_time 
    
    @property
    def trades(self) -> list[TradeRecord]:
        return self._trades 
    
    @property 
    def day_pnl(self) -> float:
        """ 당일 실현 손익(원) ─ 날짜가 바뀌면 on_bar_date()가 0으로 리셋 """
        return self._day_pnl 
    
    # ── 상태 전이:
    
    def on_bar_date(self, work_date: date) -> None:
        """ 매 봉마다 호출 ─ 날짜가 바뀌면 일일 손익을 리셋함. """
        if work_date != self._curr_date:
            self._curr_date = work_date 
            self._day_pnl = 0.0
            
    def apply_entry(self, order: Order, entry_time: datetime, buy_fee: float) -> None:
        """ 진입 반영: 포지션 등록 + 현금 차감 (체결금액 + 매수 수수로) """
        assert self._open_order is None, msg.bt.err.pf_single_position
        assert order.price is not None, msg.bt.err.pf_no_price
        
        self._open_order = order 
        self._entry_time = entry_time 
        self._cash -= order.price * order.quantity + buy_fee 
        
    def apply_exit(self, record: TradeRecord, sell_fee: float) -> None:
        """ 
        청산 반영:  현금 복원 + 거래 기록 + 일일 손익 누적 + 포지션 해제.
        현금 복원 = 진입 원금 + 총손익(gross) - 매도 수수료
                    → 진입가*수량 + (청산가-진입가)* 수량 - 매도 수수료
        """
        assert self._open_order is not None, msg.bt.err.pf_no_position
        
        # 총 손익 계산: (청산가 - 진입가) * 수량
        pnl_gross = (record.exit_price - record.entry_price) * record.quantity 
        # 현금 복원: (진입가 * 수량) + 총 손익 - 매도 수수료
        self._cash += record.entry_price * record.quantity + pnl_gross - sell_fee 
        
        self._trades.append(record)
        # TODO 0619-1344: types.TradeRecord를 포함한 전체 확인 후
        # self._day_pnl += record.pnl_net
        