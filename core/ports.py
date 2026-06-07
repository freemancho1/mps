""" 
컴포넌트 교체 비용을 낮추기 위한 Protocol 인터페이스 모음

'향후 phase가 올라갈수록 변경될 내용'은 이 인터페이스 뒤에 격리.
"""
from __future__ import annotations 

import numpy as np
from datetime import datetime  

from typing import Protocol, runtime_checkable

from mps.core.types import (
    Bar, 
    NumericInput, NumericSignal,
    PatternInput, PatternSignal,
    TradeSignal,
)


@runtime_checkable 
class DataStorePort(Protocol):
    """ 
    분봉 저장소 인터페이스
    로컬 Parquet → 시계열 DB 교체 시 인터페이스만 유지
    """
    def save_bars(self, bars: list[Bar]) -> None: ...
    def load_bars(
        self, 
        ticker: str, 
        start_date: datetime,
        end_date: datetime,
    ) -> list[Bar]: ...
    def list_tickers(self) -> list[str]: ...


# ── 모델 교체 인터피이스 (Protocol) ──────────────────
""" 
모델 포트(protocol) ─ phase 전환 시 교체 비용을 0으로 만드는 인터페이스 격리

- ThresholdModel(phase-1)·LSTMModel(phase-2)이 NumericModelPort를,
  RuleBasedPatternEngine(phase-1)·CNN1DPatternModel(phase-2)이 PatternModelPort를
  각각 구조적으로 만족함 (명시적 상속 불필요)
"""

@runtime_checkable
class NumericModelPort(Protocol):
    """ 수치 트랙 모델 인터페이스 (Threshold/LSTM/Transformer 공통) """
    def run(self, inp: "NumericInput") -> "NumericSignal": ...


@runtime_checkable
class PatternModelPort(Protocol): 
    """ 
    패턴 트랙 모델 인터페이스 (Rule/CNN/Vision 공통) 
    → 2단계 CNN부터는 "bars: list[Bar]" 부분을 사용하지 않지만 호환성을 위해 정의해둠.
    """
    def run(self, inp: "PatternInput", bars: list[Bar]) -> "PatternSignal": ...