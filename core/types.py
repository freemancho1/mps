""" 
핵심 데이터 타입 정의 ─ 모든 컴포넌트가 이 파일의 타입만 교환함

[데이터 흐름]
  Bar → (BarValidator)         → [NumericInput, PatternInput]
                                → [NumericSignal, PatternSignal]
      → (SignalAggregator)     → TradeSignal
      → (TripleBarrierGuard)   → Order
      → (PaperTrader)          → OrderResult
"""
from __future__ import annotations 

import numpy as np 
from dataclasses import dataclass 
from datetime import datetime 
from typing import Literal, Optional


Direction = Literal["BUY", "SELL", "HOLD"]
BSDirection = Literal["BUY", "SELL"]
PatternSource = Literal["RULE", "CNN", "VISION"]
OrderType = Literal["MARKET", "LIMIT"]
OrderStatus = Literal["PENDING", "FILLED", "PARTIAL", "CANCELLED"]


# ── 원시 데이터 타입 ──────────────────────────
@dataclass 
class Bar:
    """ 
    기본적인 분봉 데이터 저장 객체

    [look-ahead bias 방지가 핵심]
      - is_complete=False인 봉은 BarValidator에서 무조건 필터링됨.
      - 실 거래에서는 현재 진행중인 봉(is_complete=False)을 신호에 사용하면,
        봉이 완성되기 전 정보를 미리 쓰는 셈이 되므로 절대 허용하지 않음.
      - 백테스트에서도 동일 규칙 적용 → 합성 데이터는 is_complete=True로 생성.
    """
    ticker: str 
    timestamp: datetime         # 봉 시작 시간 (09:00 봉 → 09:00:00 KST)
    open: float 
    high: float 
    low: float 
    close: float 
    volume: int 
    is_complete: bool = False   # 봉 완성 여부 ─ False인 봉은 파이프라인에 진입 불가


