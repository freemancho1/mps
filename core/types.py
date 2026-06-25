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


# ─────────────────────────────────────
#   모델 입력
# ─────────────────────────────────────

@dataclass 
class NumericInput:
    """ 
    수치 트랙 입력. ─ NumericNormalizer의 출력
    
    - window            : 롤링 Z-score 정규화 결과.
                          - 지금 지표가 평소 대비 얼마나 차이가 있나? 
                          - 학습 기반 모델(LSTM·Transformer, Phase-2+)의 입력값.
    - raw_window        : 정규화 이전 원본 값(생성된 14개 지표값 포함)
                          = 롤 모델(ThresholdModel, Phase-1)의 입력값
    - window_size       : 이 입력 데이터를 만들 시점의 lookback 
                          · shape: window, raw_window 모두 동일
                            [lookback, n_features(=14)], dtype=float32
    """
    ticker              : str 
    timestamp           : datetime 
    window              : np.ndarray 
    raw_window          : np.ndarray 
    window_size         : int
    

@dataclass 
class PatternInput:
    """ 
    패턴 트랙 입력. ─ PatternNormalizer의 출력
    
    - ohlcv_norms       : 윈도우 내 min-max 상대 정규화 OHLCV. shape [N, 5]
    - chart_image       : 비전 모델(Phase-3+) 도입 시 사용 예약
    """
    ticker              : str 
    timestamp           : datetime 
    ohlcv_norm          : np.ndarray 
    chart_image         : Optional[np.ndarray] = None


# ─────────────────────────────────────
#   모델 출력(= 신호) + 거래 신호
# ─────────────────────────────────────

@dataclass 
class NumericSignal:
    """ 
    수치 트랙 출력(신호) → SignalAggregator의 입력
    
    - confidence        : 신호에 대한 확신도로 0.0 ~ 1.0
                          → 방향이 HOLD이면 0.0 고정
    - feature_contrib   : 피처가 이 신호에 미친 기여(영향)도
                          → 관측 가능성 충족하면서 사후 분석에 활용
    - latency_ms        : 추론 시간 ─ LatencyFilter의 판단 근거
    """
    ticker              : str 
    timestamp           : datetime 
    direction           : SignalDirection
    confidence          : float             
    feature_contrib     : dict              # {"rsi_14": 0.45, .. }  sum() = 1.0
    latency_ms          : float 
    
    
@dataclass 
class PatternSignal:
    """ 
    패턴 트랙 출력(신호) → SignalAggregator의 입력
    
    - pattern_name      : 감지된 패턴 이름 (PatternName type)
    - source            : "RULE"(phase-1), "CNN"(phase-2), "VISION"(phase-3+)
    """
    ticker              : str 
    timestamp           : datetime 
    direction           : SignalDirection
    confidence          : float 
    pattern_name        : PatternName 
    source              : PatternSource
    latency_ms          : float
    
    
@dataclass 
class TradeSignal:
    """ 
    두 트랙의 신호를 받아 합성한 신호 → RiskManager의 입력
    ─ 롱 온리이므로 방향은 항상 BUY (숏 방식은 없음)
    
    - combined_score    : 두 트랙의 가중치를 적용한 신뢰도 (활성 가중치로 정규화)
                          # TODO 0625-1531 - Config 작업 후
                          → cfg.trade
    """
    