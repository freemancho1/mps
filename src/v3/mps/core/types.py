""" 
핵심 데이터 타입 정의 ─ 모든 컴포넌트가 이 파일의 타입만을 교환함.

[데이터 흐름]
  Bar   → (BarValidator)       → [NumericInput, PatternInput]
                                → [NumericSignal, PatternSignal]
        → (SignalAggregator)   → TradeSignal
        → (BualdOrder)         → Order
        → (PaperTrader)        → OrderResult
"""
from __future__ import annotations 

import numpy as np 
from dataclasses import dataclass, field 
from datetime import datetime 
from typing import Literal, Optional, Union 


# ── 원시 데이터 ───────────────────
@dataclass 
class Bar:
    """
    가장 기본적인 분봉 데이터 저장 객체
    
    look-ahead bias를 방지하기 위해 is_complete 필드를 이용함.
    """
    ticker                  : str               
    timestamp               : datetime          # 봉 시작 시간
    open                    : float
    high                    : float 
    low                     : float
    close                   : float
    volume                  : int
    is_complete             : bool = False      # 봉 완성 여부 ─ False인 봉은 필터링(학습제외)됨.
    
    