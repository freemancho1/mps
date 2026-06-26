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
                          → cfg.risk.min_combined_score(0.55) 이상일 때만 필터 통과
    """
    ticker              : str 
    timestamp           : datetime 
    direction           : SignalDirection 
    combined_score      : float 
    numeric_confidence  : float 
    pattern_confidence  : float 
    total_latency_ms    : float


# ─────────────────────────────────────
#   주문·거래 데이터
# ─────────────────────────────────────

@dataclass 
class Order:
    """ 
    RiskManager가 승인하여 실행 레이어(PaperTrader·KISOrderClient)에 전달하는 주문.

    - stop_loss·take_profit: "체결가" 기준으로 산출한 절대 가격(원)
      → 브레이크이븐 스톱 발동 시 stop_loss는 상향 조정될 수 있음.(가변 필드)
    - expire_at: min(체결시간+time_horizon, 당일 강제청산 시각 15:15)
    - order_id: "{ticker}_{YYYYMMDD_HHMMSS}"
    - breakeven_armed: [수익성-E] + trigger 도달 여부
      → 다음 봉부터 상향된 스톱을 적용하기 위한 상태 플래그 
         (같은 봉 내 터지 순서는 알 수 없음)
    """
    order_id            : str
    order_type          : OrderType 
    ticker              : str 
    direction           : OrderAction
    take_profit         : float 
    stop_loss           : float 
    expire_at           : datetime 
    quantity            : int
    # 진입가 - submit_order 이후 채워짐
    # → 최초 주문 정보가 생성될 당시에는 이 값이 생성되지 않고,
    #    주문 정보만 생성함
    price               : Optional[float] = None
    breakeven_armed     : bool = False 


@dataclass 
class Reject:
    """ 
    RiskManager가 주문을 거부할 때 반환하는 사유 객체.

    진입 컷오프·손실 일일 한도 초과·현금 부족 등 거부 사유를 구조화해 남김
    ─ "왜 진입하지 않았는가?"도 관측 가능해야 함.
    """
    reason              : RejectReason
    signal              : TradeSignal 


@dataclass 
class OrderResult:
    """ 
    주문 체결 후 체결 결과. 
    ─ Order와 항상 쌍으로 사용되기 때문에 'ticker' 속성은 없음

    - timestamp : 백테스트에서 "시뮬레이션 시각(봉 시각)"을 주입받아 기록.
                  - 벽시계 datetime.now()를 쓰면 로그가 실제 시각축과 어긋남.
    - slippage  : 기대 가격과 체결 가격의 차이(양수 = 투자자에게 분리)
    """
    order_id            : str 
    timestamp           : datetime 
    status              : OrderStatus
    filled_price        : float         # 실 체결가
    filled_quantity     : float         # 실 체결량
    slippage            : float         # 기대 가격 대비 체결가 차이


@dataclass 
class TradeRecord:
    """ 
    완결된 거래 한 건의 기록 (진입과 청산의 쌍)
    ─ HistoricalSimulator가 청산 시 생성하여 리스트에 추가
       PerformanceEvaluator.evaluate()의 입력으로 사용

    - cost      : 왕복 "현금성 수수료" (= 매수 수수료 + 매도 수수료 + 세금)
                  → 슬리피지는 체결가에 포함되어 있으므로 여기에 포함 금지.
    - pnl_net   : 순손익(원) = (청산가 - 구매가) * 수량 - 비용(=cost)
    """
    ticker              : str 
    entry_price         : float 
    exit_price          : float 
    quantity            : int 
    entry_time          : datetime 
    exit_time           : datetime 
    exit_reason         : ExitReason 
    cost                : float 
    pnl_net             : float


@dataclass 
class PerformanceReport:
    """ 
    백테스트 성과 요약 ─ 포트폴리오 기준 정합

    - total_return_pct  : 초기 자본 대비 순손익 비율(= Σpnl_net / capital)
    - max_drawdown      : 실제 에쿼티 곡선(자본 + 청산순 누적 손익)의 최대 낙폭
    - sharpe_ratio      : "일별" 수익률 mean/std * √242.
    """
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
            f"샤프(일별): {self.sharpe_ratio:.2f}, "
            f"총 수익률(자본대비): {self.total_return_pct:.2f}, "
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