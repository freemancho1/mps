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
from dataclasses import dataclass, field
from datetime import datetime 
from typing import Literal, Optional, Union


Direction = Literal["BUY", "SELL", "HOLD"]
BSDirection = Literal["BUY", "SELL"]
PatternSource = Literal["RULE", "CNN", "VISION"]
OrderType = Literal["MARKET", "LIMIT"]
OrderStatus = Literal["PENDING", "FILLED", "PARTIAL", "CANCELLED"]
ExitReason = Literal["TAKE_PROFIT", "STOP_LOSS", "TIME_OUT", "FORCE_CLOSE"]
ExitHoldReason = Union[ExitReason, Literal["HOLD"]]


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


# ── 모델 입력용 데이터 타입 ───────────────────────

@dataclass
class NumericInput:
    """ 
    수치분석 트랙 입력 데이터 타입

    - window: NumericNormalizer가 롤링 Z-score를 적용한 결과 (결정7).
      → "지금 이 지표 값이 최근 lookback 기간 대비 몇 표준편차인가?"
      → 학습 기반 수치 모델(LSTM/Transformer, Phase-2+)의 입력.
    - raw_window: 정규화 이전 원본 지표값 (RSI 0~100, MACD 히스토그램 등).
      → Phase-1 ThresholdModel 처럼 '절대 임계값/부호'가 의미를 갖는 룰 모델이 사용.
      → Z-score는 (x-μ)/σ 반환이라 RSI 35/65 임계값과 MACD 골든·데드크로스
         부호 판정을 모두 왜곡하므로, 룰 판정에는 raw_window를 써야 함.
    - window/raw_window shape: [lookback_minutes, num_features(14)] ─ dtype=float32
    """
    ticker: str 
    timestamp: datetime 
    window: np.ndarray          # shape [N, num_features] ─ Z-score 정규화 (학습 모델용)
    raw_window: np.ndarray      # shape [N, num_features] ─ 정규화 이전 원본 (룰 모델용)
    window_size: int            # 120~240 (cfg.run.lookback_minutes=120)


@dataclass
class PatternInput:
    """ 
    패턴 분석 트랙 입력 데이터 타입

    - PatternNormalizer가 OHLCV를 윈도우 내 최소·최대 기준 0~1로 변환한 결과.
    - ohlcv_series shape: [lookbook_minutes, 5] ─ [N, (O, H, L, C, V)]
    - 절대 가격 제거 이유: 동일한 캔들 패턴이 주가 수준에 무관하게 같은 의미여야 함.
    - chart_image: 비전 모델(CNN·VLM) 도입 시 사용.
    """
    ticker: str 
    timestamp: datetime 
    ohlcv_series: np.ndarray                    # [N, 5] ─ dtype=float32
    chart_image: Optional[np.ndarray] = None    # [Height, Width, Channels] ─ Numpy Style
    

# ── 모델 출력용 신호 타입 ─────────────────────────

@dataclass 
class NumericSignal:
    """ 
    수치 트랙 ─ SignalAggregator로 전달되는 신호
    
    direction: LSTM이 판정한 방향(Phase-2)
    confidence: 모델의 확신도로 0.0 ~ 1.0
    feature_contrib: 어떤 피처가 이 신호를 만들었는지 (관측 가능성 원칙)
        → logs·signals.jsonl에 기록하여 사후 분석에 활용
    latency_ms: 모델 추론에 걸린 시간 ─ LatencyGuard의 판단 근거
    """
    ticker: str
    timestamp: datetime 
    direction: Direction
    confidence: float 
    feature_contrib: dict   # { "rsi_14": 28.3, "macd_diff": 0.12, ...}
    latency_ms: float 
    

@dataclass
class PatternSignal:
    """ 
    패턴 트랙 → SignalAggregator로 전달되는 신호

    - pattern_name: 감지된 패턴 이름 (예: "hammer", "morning_star"...)
    - source: 신호 생성 방식 ─ "RULE"(phase-1), "CNN"(phase-2), "VISION"(phase-3)
    """
    ticker: str 
    timestamp: datetime 
    direction: Direction
    confidence: float 
    pattern_name: str 
    source: PatternSource
    latency_ms: float


@dataclass 
class TradeSignal:
    """ 
    SignalAggregator의 결과로 숫자와 패턴 트랙의 합성 신호로 RiskManager의 입력으로 사용

    - 두 트랙이 같은 방향이고, 합산 지연시간이 임계값 이하이고,
      combined_score >= 0.55 일 때만 이 객체가 생성됨.
      → 이 객체는 사거나 파는 경우에만 발생하며, 가지고 있는 경우는 없음.
    """
    ticker: str 
    timestamp: datetime 
    direction: Direction
    combined_score: float 
    numeric_track_conf: float 
    pattern_track_conf: float 
    total_latency_ms: float
    

# ── 거래 정보 데이터 ─────────────────────
@dataclass 
class Order:
    """ 
    RiskManager가 승인하여 실행 레이어(PaperTrader/KISOrderClient)에 전달하는 주문.
    
    stop_loss, take_profit: Triple Barrier 기준으로 TripleBarrierGuard가 계산한 절대 가격
    expire_at: min(진입시간 + 60분, 당일 강제청산 시간(15:15))
    order_id: 백테스트에서 "{ticker}_{시간}" 문자열, 실거래에서는 KIS 주문번호로 사용
              TradeRecord에서 "entry_time"값으로 사용
    """
    ticker: str 
    direction: BSDirection
    quantity: int 
    order_type: OrderType
    stop_loss: float 
    take_profit: float 
    expire_at: datetime             # 이 시간 이후에는 자동으로 "TIMEOUT" or "FORCE_CLOSE" 처리
    order_id: str                   # 고유 주문 ID = {ticker}_{%Y%m%d}
    price: Optional[float] = None   # 진입가 (submit_order 이후 채워짐)
    
    
@dataclass
class Reject:
    """ 
    RiskManager가 주문을 거부할 때 반환하는 이유 객체
    
    Phase-1에서는 사용하지 않음.
    향후 추가 리스크 규칙(일일 손실 한도 초과 등) 구현 시 사용 예정
    """
    reason: str 
    signal: TradeSignal
    
    
@dataclass 
class OrderResult:
    """ 
    PaperTrader(or KISOrderClient)가 주문 제출 후 반환하는 체결 결과.
    
    가상 거래에서는 구현하지 않고 실거래 시 구현 예정
    """
    order_id: str 
    status: OrderStatus
    filled_price: float         # 실제 체결가
    filled_quantity: int        # 실제 체결 수량
    timestamp: datetime 
    slippage: float             # 기대가 대비 체결가 차이 (양수 = 불리하게 체결)
    

@dataclass 
class TradeRecord:
    """ 
    완결된 거래 한 건의 기록 (진입 + 청산)
    
    HistoricalSimulator가 청산 시마다 생성하여 리스트에 추가.
    PerformanceEvaluator.evaluate()의 입력으로 사용    
    """
    ticker: str 
    direction: BSDirection
    entry_price: float 
    exit_price: float 
    quantity: int 
    entry_time: str             # 진입 시 order_id 문자열 (예: "{ticker}_{%Y%m%s}")
    exit_time: object           # 청산 봉의 timestamp (datetime)
    exit_reason: ExitReason
    cost: float                 # 왕복 총 비용


# ── 성능 측정 정보 데이터 ───────────────────
@dataclass 
class PerformanceReport:
    """ 백테스트 성과 요약 보고서 """
    total_trades: int 
    win_rate: float 
    profit_factor: float 
    max_drawdown: float 
    sharpe_ratio: float 
    total_return_pct: float 
    avg_return_per_trade_pct: float 
    total_cost: float
    
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


# ── 모델 훈련 데이터 ─────────────────────

@dataclass 
class TrainHistory:
    train_loss: list[float]     = field(default_factory=list)
    val_loss: list[float]       = field(default_factory=list)
    val_acc: list[float]        = field(default_factory=list)
    best_epoch: int             = -1
    best_val_loss: float        = float("inf")
    
    
    