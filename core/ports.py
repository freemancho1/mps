""" 포트(Protocol) ─ 인터페이스 격리 계층 """
from __future__ import annotations 

from datetime import datetime 
from typing import Protocol, runtime_checkable

from mps.core.types import (
    Bar, Order, ExitHoldReason,
    NumericInput, NumericSignal, PatternInput, PatternSignal
)


@runtime_checkable
class DataStorePort(Protocol):
    """ 분봉 저장소 인터페이스: Parquet → 시계열DB """
    def save_bars(self, bars: list[Bar]) -> None: ... 
    def load_bars(
        self, 
        ticker: str, 
        start_dt: datetime, 
        end_dt: datetime,
    ) -> list[Bar]: ...


@runtime_checkable
class NumericModelPort(Protocol):
    """ 수치 트랙 모델 인터페이스 """
    def run(self, inp: NumericInput) -> NumericSignal: ... 


@runtime_checkable
class PatternModelPort(Protocol):
    """ 
    패턴 트랙 모델 인터페이스 
    
    - bars: 룰 엔진은 원본 봉(절대 가격)이 필요하고, CNN은 정규화 시계열만 사용.
            ─ 교체 가능성을 위해 시그니처를 통일 (CNN은 사용하지 않음)
    """
    def run(self, inp: PatternInput, bars: list[Bar]) -> PatternSignal: ...


@runtime_checkable 
class ExitPolicyPort(Protocol):
    """ 
    청산 정책 인터페이스 ─ 매 봉마다 주문을 검사해 청산 사유(또는 "HOLD")를 반환
    """
    def check(self, order: Order, bar: Bar) -> ExitHoldReason: ...
