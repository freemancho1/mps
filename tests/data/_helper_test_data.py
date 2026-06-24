""" data 테스트 공용 헬퍼 """
from __future__ import annotations 

import numpy as np 
import pytest 
from datetime import datetime, timezone 

from mps.core.config import cfg
from mps.core.types import Bar 


def make_bars(
    n_bars: int, 
    base_price: float = 10_000.0,
    seed: int = cfg.sys.seed
) -> list[Bar]:
    """ 재현 가능한 합성봉 n개 생성 """
    rng = np.random.default_rng(seed)
    bars: list[Bar] = []
    price = base_price
    for i in range(n_bars):
        close = max(price * (1 + rng.normal(0.0, 0.005)), 1.0)
        high = close * (1 + abs(rng.normal(0, 0.002)))
        low = close * (1 - abs(rng.normal(0, 0.002)))
        bars.append(Bar(
            ticker="000000",
            timestamp=datetime(2025, 1, 5, 9, i%60, tzinfo=cfg.sys.timezone),
            open=round(price, 0),
            high=round(high, 0),
            low=round(low, 0),
            close=round(close, 0),
            volume=int(rng.integers(100, 10_000)),
            is_complete=True,
        ))
        price = close 
        
    return bars


def make_flat_bars(n_bars: int, price: float = 10_000.0) -> list[Bar]:
    """ OHLC 전부 동일한 값 (분산 0), V=1000 """
    return [
        Bar(
            ticker="000000",
            timestamp=datetime(2025, 1, 5, 9, i%60, tzinfo=cfg.sys.timezone),
            open=price, high=price, low=price, close=price,
            volume=1_000, is_complete=True,
        ) 
        for i in range(n_bars)
    ]
    
    
@pytest.fixture 
def bars_200() -> list[Bar]:
    return make_bars(200)
    
