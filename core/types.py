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
from typing import Literal, Optional, Union, TypeAlias


# ─────────────────────────────────────
#   별칭 정의
# ─────────────────────────────────────
DataSource      : TypeAlias = Literal["KIS", "PYKRX", "STORE"]
# 신호·라벨 방향: 매수 후보(BUY) 또는 관망(HOLD)
SignalDirection : TypeAlias = Literal["BUY", "HOLD"]
# 체결 행위(주문·청산 방향): 매수(진입, BUY), 매도(청산, SELL)
OrderAction     : TypeAlias = Literal["BUY", "SELL"]

# Track 유형
TrackType       : TypeAlias = Literal["numeric", "pattern"]

# 패턴 신호 추적 (phase 추적)
PatternSource   : TypeAlias = Literal["RULE", "CNN", "VISION"]
PatternName     : TypeAlias = Literal[
                    "HAMMER", "SHOOTING_STAR", "BULLISH_ENGULFING", "BEARISH_ENGULFING",
                    "MORNING_STAR", "EVENING_STAR", "BOX_BREAKOUT_UP", "BOX_BREAKOUT_DOWN",
                    "CNN_SEQ",
                    "NONE"
                ]

# 두 트랙 신호 결합 정책
# ─ "confluence": 두 트랙 모두 BUY일 때만 진입 (AND 게이트, 구 require_confluence=True)
# ─ "weighted"  : 한쪽만 BUY여도 진입 허용 + 두 트랙 합의 시 confluence_bonus 가산
AggregationMode : TypeAlias = Literal["CONFLUENCE", "WEIGHTED"]

OrderType       : TypeAlias = Literal["MARKET", "LIMIT"]
OrderStatus     : TypeAlias = Literal["PENDING", "FILLED", "PARTIAL", "CANCELLED"]
ExitReason      : TypeAlias = Literal["TAKE_PROFIT", "STOP_LOSS", "TIME_OUT", "FORCE_CLOSE"]
ExitHoldReason  : TypeAlias = Union[ExitReason, Literal["HOLD"]]
RejectReason    : TypeAlias = Literal["ENTRY_CUTOFF", "DAILY_LOSS_LIMIT", "NO_CASH"]


# ─────────────────────────────────────
#   원시 데이터
# ─────────────────────────────────────
@dataclass 
class Bar:
    """ 
    분봉 1개 정의 ─ 시스템 전체의 기본 입력 단위 
    
    look-ahead bias를 방지하기 위해 is_complete 필드를 이용함.
    """
    ticker              : str 
    timestamp           : datetime 
    open                : float 
    high                : float 
    low                 : float 
    close               : float 
    volume              : int 
    is_complete         : bool = False


