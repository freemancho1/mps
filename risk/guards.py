""" 
리스크 가드 모음 — 모델과 완전히 분리된 독립 가드레일.

[설계 원칙]
- 모델이 얼마나 확신에 차 있더라도 리스크 가드를 우회할 수 없다.
- 가드는 모델의 출력을 신뢰하지 않는다 — 언제나 독립적으로 작동한다.

[세 가지 가드]
- TripleBarrierGuard: 주문 생성 시 TP/SL/만료 기준가 설정
- IntradayCloseoutGuard: 장 마감 강제청산 여부 판단
- StopLossTakeProfitGuard: 매 봉마다 보유 포지선의 청산 조건 체크
"""
from __future__ import annotations 

from datetime import datetime, timedelta 
from typing import Literal 

from mps.data.types import Order, TradeSignal
from mps.sys.config import settings
from mps.data.calendar import force_close_dt


class TripleBarrierGuard:
    """ 
    Triple Barrier 기반 주문 생성

    [Triple Barrier 란]
    - 진입 이후 세 가지 장벽(barrier) 중 하나를 먼저 터치하면 청산
      · 상단 장벽(+0.5%): 익절
      · 하단 장벽(-0.3%): 손절
      · 시간 장벽(60분): 시간 만료 청산
    - 비대칭 설정(TP=0.5% > SL=0.3%)의 이유
      · 손실이 이익보다 더 빨리 차단 → 승률이 낮아도 수익 기대값 유지 가능.
    
    [만료 시각]
    - min(진입 + 60분, 당일 강제 청산 시각 (15:15))
      → 60분이 지나도 15:15 이전이면 15:15에 강제 청산.
    """