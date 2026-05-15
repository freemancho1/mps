""" 
BarValidator - look-ahead bias 방지 경계선.

[핵심 원칙]
· 09:31:00 시점에는 09:30봉이 완성(is_complete=True)된 이후에만 신호를 생성할 수 있다.
· 현재 진행중인 봉(is_complete=False)의 데이터를 신호로 사용하면,
  "아직 일어나지 않은 미래"의 정보를 쓰는 것이 되므로 엄격히 금지.

[실거래 vs 백테스트]
· 실거래: WebSocket에서 봉 완성 이벤트(is_complete=True)를 받을 때만 파이프라인 진입.
· 테스트: 합성 봉인 모두 is_complete=True로 생성되므로 사실상 전부 통과.
         실제 틱 데이터 사용 시 is_complete=False 봉이 섞일 수 있어 이 필터가 중요

※ 이 클래스는 단순하지만 전체 시스템이 동작하는 동안 절대로 깨지면 안되는 규칙을 가짐
   - 규칙: "미 완성 봉(is_complete=False)은 데이터로 사용하지 않음"
"""
from __future__ import annotations 

from mps.data.types import Bar 


class BarValidator:

    def validate(self, bar: Bar) -> bool:
        """ 완성된 봉이면 True, 진행 중인 봉이면 False. """
        return bar.is_complete
    
    def filter(self, bars: list[Bar]) -> list[Bar]:
        """ 미완성 봉(is_complete=False)을 제거하고 완성 봉 리스트만 반환. 
        
            HistoricalSimulator.run()의 첫 번째 단계로 호출.
            필터링 후 봉 수가 lookback + 1 미만이면 simulator가 ValueError 발생
        """
        return [b for b in bars if b.is_complete]