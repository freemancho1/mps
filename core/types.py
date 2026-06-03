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
    timestampe: datetime 
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
    

# ── 모델 훈련 데이터 ─────────────────────

@dataclass 
class TrainHistory:
    train_loss: list[float]     = field(default_factory=list)
    val_loss: list[float]       = field(default_factory=list)
    val_acc: list[float]        = field(default_factory=list)
    best_epoch: int             = -1
    best_val_loss: float        = float("inf")
    
    
    