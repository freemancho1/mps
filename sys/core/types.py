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
    
    - NumerialNormalizer가 feature_matrix에 롤링 Z-score를 적용한 결과임.
    - window shape: [lookback_minutes, num_features(14)]
    - 각 값은 "지금 이 지표 값이 최근 lookback 기간 대비 몇 표준편차인가?"를 나타냄.
      → 절대값이 아닌 상대적 비정상도를 모델에 입력하므로 종목·시기에 무관하게 일반화.
    """
    ticker: str                 
    timestamp: datetime
    window: np.ndarray          # shape [N, num_features] ─ dtype=float32
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
