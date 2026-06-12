""" 
핵심 데이터 타입 정의 ─ 모든 컴포넌트가 이 파일의 타입만을 교환함.

[설계 원칙]
  - 컴포넌트들은 "이 파일의 타입만" 주고 받음.
    → 어떤 구현체(룰·LSTM·CNN, Paper·KIS)로 교체해도 이 원칙만 지키면 됨.
  - 이 파일은 다른 mps 모듈을 import하지 안음(의존성 최하층, cfg보다 아래)

[데이터 흐름]
  Bar   → (BarValidator)       → [NumericInput, PatternInput]
        → (모델 추론)           → [NumericSignal, PatternSignal]
        → (Aggregator)         → TradeSignal
        → (RiskManager)        → Order | Reject
        → (PaperTrader)        → OrderResult ─ (청산 시) → TradeRecord
        → (Evaluator)          → PerformanceReport
"""
from __future__ import annotations 

import numpy as np 
from dataclasses import dataclass, field 
from datetime import datetime 
from typing import Literal, Optional, Union 


# ─────────────────────────────────────
#   별칭 정의
# ─────────────────────────────────────
# 신호·라벨 방향: 매수 후보(BUY) 또는 관망(HOLD) 
# ─ 신호 차원에 SELL(매도)는 존재하지 안음 → 매도는 가드를 이용해 처리
SignalDirection = Literal["BUY", "HOLD"]
# 체결 행위(주문 방향): 매수(진입, BUY), 매도(청산, SELL)
# ─ 신호의 방향과 체결 행위를 타입 수준에서 구분해 혼용을 차단
OrderAction = Literal["BUY", "SELL"]

# 패턴 신호 추적 (phase 추적)
PatternSource = Literal["RULE", "CNN", "VISION"]
OrderType = Literal["MARKET", "LIMIT"]
OrderStatus = Literal["PENDING", "FILLED", "PARTIAL", "CANCELLED"]
ExitReason = Literal["TAKE_PROFIT", "STOP_LOSS", "TIME_OUT", "FORCE_CLOSE"]
ExitHoldReason = Union[ExitReason, Literal["HOLD"]]


# ─────────────────────────────────────
#   원시 데이터
# ─────────────────────────────────────
@dataclass 
class Bar:
    """
    본봉 1개 정의 ─ 시스템 전체의 기본 입력 단위
    
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
    
    