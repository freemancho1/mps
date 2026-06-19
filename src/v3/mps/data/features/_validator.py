""" 
BarValidator ─ look-ahead bias 방지의 최전선

[역할]
  - is_complete=False: 현재 생성(형성)중인 봉을 파이프라인에서 차단.
  - 실거래: 진행 중인 봉의 형태를 패턴 판정에 쓰면 봉 완성 전 정보를 미리 쓰는 셈
    → 절대 허용할 수 없는 데이터 누수
  - 백테스트: 동일 규칙 적용. 합성 데이터는 is_complete=True로 생성됨.
"""
from __future__ import annotations 

from mps.core.types import Bar 


class BarValidator:
    def validator(self, bar: Bar) -> bool:
        """ 단일 봉 검증 ─ 완성 봉만 통과 """
        return bar.is_complete
    
    def filter(self, bars: list[Bar]) -> list[Bar]:
        """ 완성 봉만 남긴 새 리스트 반환 (원본 불변) """
        return [bar for bar in bars if bar.is_complete]