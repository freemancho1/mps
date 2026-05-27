""" 
BarValidator ─ look-ahead bias 방지 경계선.

[핵심 원칙]
  - 09:31:00 시점에는 09:30봉이 완성(is_complete=True)된 이후에만 신호를 생성할 수 있음.
  - 현재 진행중인 봉(is_complete=False)의 데이터를 신호에 사용하면
    "아직 일어나지 않는 미래"의 정보를 쓰는 것이므로 엄격히 금지.
    
[실거래 vs 백테스트]
  - 실거래: WebSocket에서 봉 완성 이벤트(is_complete=True)를 받을 때만 파이프라인 진입.
  - 백테스트: 합성 봉은 모두 is_complete=True로 생성되므로 사실상 전부 통과.
              ⇒ 실 데이터 사용 시 is_complete=False 봉이 섞일 수 있기 때문에 이 필터가 중요

※ 이 클래스는 아주 간단하지만, 
   전체 시스템의 원칙─ 미래 데이터는 학습에 사용하지 않는다 ─을 지키는 중요한 클래스임.
"""
from __future__ import annotations 

from mps.sys.core.types import Bar 


class BarValidator: 
    def validate(self, bar: Bar) -> bool:
        # 완성된 봉이면 True, 진행중인 봉이면 False를 리턴
        return bar.is_complete

    def filter(self, bars: list[Bar]) -> list[Bar]:
        # 미완성 봉을 제거하고 완성 봉만 반환
        return [bar for bar in bars if bar.is_complete]