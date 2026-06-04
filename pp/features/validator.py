""" 
BarValidator ─ look-ahead bias 방지 경계선
"""
from __future__ import annotations

from mps.core.types import Bar 


class BarValidator:
    def validate(self, bar: Bar) -> bool:
        return bar.is_complete
    
    def filter(self, bars: list[Bar]) -> list[Bar]:
        return [bar for bar in bars if bar.is_complete]