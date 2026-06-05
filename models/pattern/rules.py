""" 
Phase-1 Pattern Model ─ RuleBasedPatternEngine: 캔들스틱·차트 패탄 룰 기반 판정

[7가지 패턴, 우선순위 순]
  - 단봉: hammer(망치형), shooting_star(유성형)
  - 이중봉: bullish_engulfing(강세 장악형), bearish_engulfing(약세 장악형)
  - 삼봉: morning_star(샛별형), evening_star(저녁별형)
  - 차트: box_breakout(박스권 돌파)
  
[우선순위 정책]
  - 첫 번째 매칭에서 즉시 반환 (Break-first)
  - 삼봉 패턴이 단봉 패턴보다 우선순위가 낮은 이유:
    → 현재 코드에서 단봉이 먼저 검사되므로 단봉이 우선시 됨.
    → 삼봉 패턴이 더 강력하지만 드물기 때문에 Phase-1에서는 의도적으로 단봉을 우선함.
    
[교체 계획]
  - Phase-2: CNN 기반 차트 이미지 인식으로 일부 대체.
  - source 필드를 "RULE" → "CNN"으로 변경하면 상위 레이어 변경 없이 통합 가능.
"""
from __future__ import annotations 

import time 

from mps.config import cfg 
from mps.core.types import Bar, Direction, PatternInput, PatternSignal


class RuleBasedPatternEngine:

    def _detect(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """ 
        아래 우선순위 순으로 패턴을 검사하고 첫 번째 매칭을 그래도 반환
        
        - bars는 정규화되지 않는 원본 리스트를 사용 (절대 가격 사용: 비율 계산에 필요)
        """
        if len(bars) < 5:
            return cfg.run.no_signal_pattern
        
        # TODO Z: 여기서부터 실행
        
    # ── 체크함수 정의 ────────────────────
    def _hammer(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        pass 
    
    def _shooting_star(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        pass 
    
    def _bullish_engulfing(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        pass 
    
    def _bearish_engulfing(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        pass 
    
    def _morning_star(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        pass 
    
    def _evening_star(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        pass 
    
    def _box_breakout(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        pass 
