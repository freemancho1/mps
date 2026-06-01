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
from dataclasses import dataclass 
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

    
