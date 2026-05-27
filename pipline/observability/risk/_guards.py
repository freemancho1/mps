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
        
    """