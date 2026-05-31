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

