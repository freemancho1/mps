""" 
핵심 데이터 타입 정의 ─ 모든 컴포넌트가 이 파일의 타입만 교환함.

[데이터 흐름]
  Bar   → (BarValidator)        → [NumericalInput, PatternInput]
                                 → [NumericalSignal, PatternSignal]
        → (SignalAggregator)    → TradeSignal
        → (TripleBarrierGuard)  → Order
        → (PaperTrader)         → OrderResult
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


# ── 원시 데이터 타입 ─────────────────────────────
@dataclass 
class Bar:
    """ 
    기본 분봉 데이터 저장 객체
    
    [look-ahead bias 방지가 핵심]
      - is_complete=False인 봉은 BarValidator에서 무조건 필터링
      - 실 거래에서는 현재 진행중인 봉(is_complete=False)을 신호에 사용하면,
        봉이 완성되기 전 정보를 미리 쓰는 셈이 되므로 절대 허용하지 않음.
      - 백테스트에서도 동일 규칙 적용 → 합성 데이터는 is_complete=True로 생성.
    """
    ticker: str                 # KRX 종목 코드 (예: "005930")
    timestamp: datetime         # 봉 시작 시간 (09:00 봉 → 09:00:00 KST)
    open: float 
    high: float 
    low: float 
    close: float 
    volume: int 
    is_complete: bool = False   # 봉 완성 여부 ─ False인 봉은 파이프라인에 진입 불가    
    

# ── 모델 입력용 데이터 타입 ─────────────────────────
@dataclass 
class NumericalInput:
    """ 
    수치분석 트랙 입력 데이터 타입
    
    - window: NumerialNormalizer가 롤링 Z-score를 적용한 결과 (결정7).
      → "지금 이 지표 값이 최근 lookback 기간 대비 몇 표준편차인가?"
      → 학습 기반 수치 모델(LSTM/Transformer, Phase-2+)의 입력.
    - raw_window: 정규화 이전 원본 지표값 (RSI 0~100, MACD 히스토그램 등).
      → Phase-1 ThresholdModel처럼 '절대 임계값/부호'가 의미를 갖는 룰 모델이 사용.
      → Z-score는 (x-μ)/σ 변환이라 RSI 35/65 임계값과 MACD 골든/데드크로스
         부호 판정을 모두 왜곡하므로, 룰 판정에는 raw_window를 써야 한다.
    - window/raw_window shape: [lookback_minutes, num_features(14)] ─ dtype=float32
    """
    ticker: str                 
    timestamp: datetime
    window: np.ndarray          # shape [N, num_features] ─ Z-score 정규화 (학습 모델용)
    raw_window: np.ndarray      # shape [N, num_features] ─ 정규화 이전 원본 (룰 모델용)
    window_size: int            # 120~240 (cfg.sys.lookback_minutes=120)
    

@dataclass 
class PatternInput: 
    """ 
    패턴분석 트랙 입력 데이터 타입
    
    - PatternNormalizer가 OHLCV를 윈도우 내 최소·최대 기준 0~1로 변환한 결과.
    - ohlcv_series shape: [lookback_minutes, 5] ─ [N, (Open, High, Low, Close, Volume)]
    - 절대 가격 제거 이유: 동일한 캔들 패턴이 주가 수준에 무관하게 같은 의미여야 하므로.
    - chart_image: 비전 모델(CNN/VLM) 도입 시 사용. Phase-1에서는 사용하지 않음.
    """
    ticker: str 
    timestamp: datetime 
    ohlcv_series: np.ndarray                    # shape [N, 5] ─ dtype=float32
    chart_image: Optional[np.ndarray] = None    # shape [H, W, C] ─ Phase-1에서는 사용 않함


# ── 모델 출력 신호 타입 ─────────────────────────

@dataclass 
class NumericalSignal:
    """ 
    수치 트랙 → SignalAggregator로 전달되는 신호

    direction: ThresholdModel(Phase-1) 또는 LSTM(Phase-2+)이 판정한 방향
    confidence: 모델의 확신도로 0.0 ~ 1.0
    feature_contrib: 어떤 피처가 이 신호를 만들었는지 (관측 가능성 원칙)
        logs·signals.jsonl에 기록하여 사후 분석에 활용
    latency_ms: 모델 추론에 걸린 시간 ─ LatencyGuard의 판단 근거
    """
    ticker: str 
    timestamp: datetime 
    direction: Direction
    confidence: float 
    feature_contrib: dict   # { "rsi_14": 28.3, "macd_diff": 0.12...}
    latency_ms: float 


@dataclass 
class PatternSignal:
    """
    패턴 트랙 → SignalAggregator로 전달되는 신호

    pattern_name: 감지된 패턴 이름 (예: "hammer", "morning_str"...)
    source: 신호 생성 방식 ─ "RULE"(Phase-1), "CNN"(Phase-2), "VISION"(Phase-3)
    """
    ticker: str 
    timestamp: datetime 
    direction: Direction
    confidence: float 
    pattern_name: str 
    source: Literal["RULE", "CNN", "VISION"]
    latency_ms: float 


@dataclass 
class TradeSignal:
    """ 
    SignalAggregator → RiskManager로 전달되는 합의 신호

    두 트랙이 같은 방향에 동의했고, 합산 지연시간이 임계값 이하이고,
    combined_score >= 0.55 일 때만 이 객체가 생성됨.
    → 이 객체는 사거나 파는 경우만 발생하며, 가지고 있는 경우는 없음.

    combined_score 계산식:
      - 두 트랙 가중치 합산: num_conf * 0.5(가중치) + pat_conf * 0.5(가중치)
    """
    ticker: str 
    timestamp: datetime 
    direction: BSDirection
    combined_score: float 
    num_track_conf: float 
    pat_track_conf: float 
    total_latency_ms: float 


@dataclass 
class Order:
    """ 
    RiskManager가 승인하여 실행 레이어(PaperTrader/KISOrderClient)에 전달하는 주문.

    stop_loss, take_profig: Triple Barrier 기준으로 TripleBarrierGuard가 계산한 절대 가격
    expire_at: min(진입시간 + 60분, 당일 강제청산 시간(15:15))
    order_id: 백테스트에서 "{ticker}_{시각}" 문자열, 실거래에서는 KIS 주문번호
              TradeRecord의 entry_time 값으로 사용
    """
    ticker: str 
    direction: BSDirection
    quantity: int 
    order_type: OrderType
    stop_loss: float                # 손절 기준가 (절대 가격, 원 단위)
    take_profit: float              # 익절 기준가 (절대 가격, 원 단위)
    expire_at: datetime             # 이 시각 이후 봉에서 TIMEOUT 또는 FORCE_CLOSE 처리
    price: Optional[float] = None   # 진입가 (submit_order 이후 채워짐)
    order_id: Optional[str] = None  # 고유 주문 id
    

@dataclass 
class Reject:
    """ 
    RiskManager가 주문을 거부할 때 반환하는 이유 객체
    
    현재 Phase-1에서 Reject를 발생시키는 명시적 규칙이 없지만,
    (수량=0 이면 주문 자체를 건너뜀)
    향후 추가 리스크 규칙(일일 손실 한도 초과 등) 구현 시 사용.
    """
    reason: str 
    signal: TradeSignal
    

@dataclass 
class OrderResult:
    """ 
    PaperTrader(또는 KISOrderClient)가 주문 제출 후 반환하는 체결 결과.
    
    Phase-1(PaperTrader): 항상 FILLED ─ 부분 체결·거부 미 시뮬레이션.
                          → 실거래 전환 시 PARTIAL, CANCELLED 케이스도 처리해야 함.
    
    slippage: filled_price - 주문 당시 시장가 (BUY는 양수, SELL은 음수)
              → CostModel의 slippage_rate와 벌개로 기록 (실제 슬리피지 모니터링)
    """
    order_id: str 
    status: OrderStatus
    filled_price: float         # 실제 체결가
    filled_quantity: int 
    timestamp: datetime 
    slippage: float             # 기대가 대비 체결가 차이 (양수 = 불리하게 체결)