""" BarValidator ─ look-ahead bias 방지 """
from __future__ import annotations 

from mps.core.types import Bar 


class BarValidator:
    
    @staticmethod
    def validator(bar: Bar) -> bool:
        """ 단일 봉 검증 ─ 완성 봉만 통과 """
        return bar.is_complete
    
    @staticmethod
    def filter(bars: list[Bar]) -> list[Bar]:
        """ 완성 봉만 남긴 새 리스트 반환 (원본 불편) """
        return [bar for bar in bars if bar.is_complete]