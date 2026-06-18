""" 
포트(protocol) ─ 인터페이스 격리 계층
"""
from __future__ import annotations 

from datetime import datetime 
from typing import Protocol, runtime_checkable

from mps.core.types import Bar, Order, ExitHoldReason
from mps.core.types import NumericInput, NumericSignal, PatternInput, PatternSignal


@runtime_checkable
class DataStorePort(Protocol):
    """ 
    분봉 저장소 인터페이스 
    ─ 로컬 Parquet(현재) → 시계열 DB(예정) 교체 시 이 계약만 유지
    """
    def save_bars(self, bars: list[Bar]) -> None: ...
    def load_bars(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime,
    ) -> list[Bar]: ...
    
    
@runtime_checkable
class NumericModelPort(Protocol):
    """ 
    수치 트랙 모델 인터페이스.
    ThresholdModel(룰)·LSTMModel(학습)
    """