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
DataSource = Literal["KIS", "PYKRX", "STORE"]
# 신호·라벨 방향: 매수 후보(BUY) 또는 관망(HOLD) 
# ─ 신호 차원에 SELL(매도)는 존재하지 안음 → 매도는 가드를 이용해 처리
SignalDirection = Literal["BUY", "HOLD"]
# 체결 행위(주문 방향): 매수(진입, BUY), 매도(청산, SELL)
# ─ 신호의 방향과 체결 행위를 타입 수준에서 구분해 혼용을 차단
OrderAction = Literal["BUY", "SELL"]

# 패턴 신호 추적 (phase 추적)
PatternSource = Literal["RULE", "CNN", "VISION"]
PatternName = Literal[
    "HAMMER", "SHOOTING_STAR", "BULLISH_ENGULFING", "BEARISH_ENGULFING",
    "MORNING_STAR", "EVENING_STAR", "BOX_BREAKOUT_UP", "BOX_BREAKOUT_DOWN",
    "CNN_SEQ",
    "NONE"
]
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
    ticker              : str               
    timestamp           : datetime          # 봉 시작 시간
    open                : float
    high                : float 
    low                 : float
    close               : float
    volume              : int
    is_complete         : bool = False      # 봉 완성 여부 ─ False인 봉은 필터링(학습제외)됨.
    

# ─────────────────────────────────────
#   모델 입력
# ─────────────────────────────────────

@dataclass 
class NumericInput:
    """ 
    수치 트랙 입력.

    - window            : 롤링 Z-score 정규화 결과.
                          - "지금 지표가 평소 대비 몇 표준편차(σ)인가?"
                          - 학습 기반 모델(LSTM·Transformer)의 입력값.
    - raw_window        : 정규화 이전 원본 지표값 (RSI 0~100, MACD 히스토그램 등)
                          - 절대 임계값(RSI-35·65)·부호(골든·데드크로스)가 의미를 갖는
                            룰 모델(ThresholdModel)이 사용. Z-score는 이 값이 왜곡됨.
    - shape             : [lookback_minutes, num_features(=14)], dtype=float32
    """
    ticker              : str 
    timestamp           : datetime 
    window              : np.ndarray 
    raw_window          : np.ndarray 
    window_size         : int 


@dataclass 
class PatternInput:
    """ 
    패턴 트랙 입력.

    - ohlcv_series      : 윈도우 내 min-max 상대 정규화 OHLCV. shape [N, 5]
                          → 상대가격으로 변경 이유는 20만원대 망치형과 5만원대 망치형이 동일하기 때문
                             (같은 형태는 가격대와 무관하게 같은 특징 벡터를 가져야 함)
    - chart_image       : 비전 모델(Phase-3) 도입 시 사용 예약                             
    """
    ticker              : str 
    timestamp           : datetime 
    ohlcv_series        : np.ndarray 
    chart_image         : Optional[np.ndarray] = None 


# ─────────────────────────────────────
#   모델 출력(= 신호) + 거래 신호
# ─────────────────────────────────────

@dataclass 
class NumericSignal:
    """ 
    수치 트랙 출력(신호) → SignalAggregator의 입력

    - feature_contrib   : 어떤 피처가 이 신호를 만들었는지 (관측 가능성 원칙 충족)
                          → signals.jsonl에 기록되어 사후 분석에 사용
    - latency_ms        : 추론 시간 ─ LatencyFilter의 판단 근거
    """
    ticker              : str 
    timestamp           : datetime 
    dir                 : SignalDirection
    confidence          : float             # 0.0~1.0 (HOLD는 0.0 고정)
    feature_contrib     : dict              # {"rsi_14": 28.3, "macd_diff": 0.12...} sum=1.0
    latency_ms          : float


@dataclass 
class PatternSignal:
    """ 
    패턴 트랙 출력(신호) → SignalAggregator의 입력

    - pattern_name      : 감지된 패턴의 이름 ("HAMMER", "CNN_REQ" 등)
    - source            : "RULE"(Phase-1), "CNN"(Phase-2), "VISION"(Phase-3)
    """
    ticker              : str 
    timestamp           : datetime 
    dir                 : SignalDirection
    confidence          : float 
    pattern_name        : str 
    source              : PatternSource
    latency_ms          : float 


@dataclass 
class TradeSignal:
    """ 
    두 트랙의 신호를 받아 합성한 신호 → RiskManager의 입력 신호로 감.
    ─ 롱 온리이므로 방향은 항상 "BUY"

    - combined_score    : 기여 트랙들의 가중 평균 신뢰도(활성 가중치로 정규화)
                          ─ cfg.trade.min_combined_score (=0.47) 이상일 때만 필터를 통과함
    """
    ticker              : str 
    timestamp           : datetime 
    dir                 : SignalDirection
    combined_score      : float 
    numeric_track_conf  : float 
    pattern_track_conf  : float 
    total_latency_ms    : float 


# ─────────────────────────────────────
#   주문·거래 데이터
# ─────────────────────────────────────

# ── 주문 정보
@dataclass 
class Order:
    """ 
    RiskManager가 승인하여 실행 레이어(PaperTrader·KISOrderClient)에 전달하는 주문.
    
    - stop_loss, take_profit: TripleBarrier 기준으로 TripleBarrierGuard가 계산한 절대 가격
    - expire_at: min(진입시간 + 60분(time_horizon), 당일 강제청산 시간(15:15))
    - order_id: 백테스트에서 "{ticker}_{일시}" 문자열 → "%Y%m%d%H%M%S"
                실거래에서는 KIS 주문번호로 사용하고 TradeRecord에서 "entry_time"값으로 활용
    """
    ticker              : str
    dir                 : OrderAction
    quantity            : int
    order_type          : OrderType
    stop_loss           : float 
    take_profit         : float 
    expire_at           : datetime 
    order_id            : str
    # 진입가 ─ submit_order 이후 채워짐
    # - 최초 주문 정보가 생성될 당시에는 이 값이 생성되지 않고, 주문 정보만 생성됨.
    price               : Optional[float] = None    
    

@dataclass 
class Reject:
    """ 
    RiskManager가 주문을 거부할 때 반환하는 이유 객체
    - Phase-1에서는 사용하지 않지만,
      향후 추가 리스크 규칙(일일 손실 한도 초과 등) 구현 시 사용 예정
    """
    reason              : str
    signal              : TradeSignal 
    
    
@dataclass 
class OrderResult:
    """ 
    PaperTrader(or KISOrderClient)가 주문 제출 후 반환하는 체결 결과
    - 가상 거래에서는 구현하지 않고 실거래 시 구현 예정
    """
    order_id            : str
    timestamp           : datetime
    status              : OrderStatus
    filled_price        : float         # 실 체결가
    filled_quantity     : int           # 실 거래 수량
    slippage            : float         # 기대가 대비 체결가 차이 (양수 = 불리하게 체결)
    
    
@dataclass 
class TradeRecord:
    """ 
    완결된 거래 한 건의 기록 (진입과 청산의 쌍)
    - HistoricalSimulator가 청산 시 마다 생성하여 리스트에 추가
    - PerformanceEvaluator.evaluate()의 입력으로 사용
    """
    ticker              : str 
    dir                 : OrderAction
    entry_price         : float 
    exit_price          : float 
    quantity            : int 
    entry_time          : str           # 진입 시 order_id 문자열("{ticker}_{일시}") 사용
    exit_time           : object        # 청산 봉의 timestamp(= datetime)
    exit_reason         : ExitReason
    cost                : float         # 왕복 총 비용

# ── 거래 성능 측정
@dataclass 
class PerformanceReport:
    """ 백테스트 성과 요약 보고서 """
    total_trades        : int 
    win_rate            : float 
    profit_factor       : float 
    max_drawdown        : float 
    sharpe_ratio        : float 
    total_return_pct    : float 
    avg_return_per_trade_pct: float 
    total_cost          : float 
    
    def __str__(self) -> str:
        return (
            f"총 거래: {self.total_trades:,}건, "
            f"승률: {self.win_rate:.1%}, "
            f"수익 인수: {self.profit_factor:.2f}, "
            f"최대 낙폭: {self.max_drawdown:.1%}, "
            f"샤프: {self.sharpe_ratio:.2f}, "
            f"총 수익률: {self.total_return_pct:.2f}, "
            f"거래당 수익률: {self.avg_return_per_trade_pct:.4%}, "
            f"총 비용: {self.total_cost:,.0f}원"            
        )
        

# ─────────────────────────────────────
#   모델 학습 
# ─────────────────────────────────────
@dataclass 
class TrainHistory:
    """ 학습 곡선·조기 종료 추적 (재현 가능성: 결과와 함께 기록 권장) """
    train_loss          : list[float]   = field(default_factory=list)
    val_loss            : list[float]   = field(default_factory=list)
    val_acc             : list[float]   = field(default_factory=list)
    best_epoch          : int           = -1
    best_val_loss       : float         = float("inf")

