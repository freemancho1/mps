""" 
RuleBasedPatternEngine ─ 캔들스틱·차트 패턴 룰 기반 판정.

[7가지 패턴, 우선순위 순]
  - 단봉: hammer(망치형), shooting_star(유성형)
  - 이중봉: bullish_engulfing(강세 장악형), bearish_engulfing(약세 장악형)
  - 삼봉: morning_star(샛별형), evening_star(저녁별형)
  - 차트: box_breakout(박스권 돌파)
  
[우선순위 정책]
  - 첫 번째 매칭에서 즉시 반환 (Break-first)
  - 삼봉 패턴이 단봉 패턴보다 우선순위가 낮은 이유:
    → 현재 코드에서는 단봉이 먼저 검사되므로 단봉이 우선됨.
    → 이는 Phase-1에서 의도적 선택으로 삼봉 패턴이 더 강력하지만 드물기 때문임
    
[교체 계획]
  - Phase-2: CNN 기반 차트 이미지 인식으로 일부 대체.
  - source 필드를 "RULE" → "CNN"으로 변경하면 상위 레이어 변경 없이 통합 가능
"""
from __future__ import annotations 

import time 
import numpy as np
from datetime import datetime 

from mps.sys.core.types import Bar, Direction, PatternInput, PatternSignal
from mps.sys import cfg


class RuleBasedPatternEngine: 
    
    _NO_SIGNAL: tuple[Direction, float, str] = ("HOLD", 0.0, "none")
    
    def _detect(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """ 
        우선순위 순서로 패턴을 검사하고 첫 번째 매칭을 그대로 반환
        
        bars는 원본 Bar 리스트 (절대 가격 사용: 비율 계산에 필요)
        """
        if len(bars) < 5:
            return self._NO_SIGNAL
        
        checks_fn = [
            self._hammer, 
            self._shooting_star,
            self._bullish_engulfing, 
            self._bearish_engulfing,
            self._morning_star,
            self._evening_star,
            self._box_breakout,
        ]
        for fn in checks_fn:
            direction, confidence, name = fn(bars)
            if direction != "HOLD":
                return direction, confidence, name 
            
        return self._NO_SIGNAL
    
    def run(self, inp: PatternInput, bars: list[Bar]) -> PatternSignal:
        """ 패턴 감지 후 PatternSignal 반환. 지연시간 측정 포함. """
        start_time = time.perf_counter()
        direction, confidence, pattern_name = self._detect(bars)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return PatternSignal(
            ticker=inp.ticker,
            timestamp=inp.timestamp,
            direction=direction,
            confidence=confidence,
            pattern_name=pattern_name,
            source="RULE",
            latency_ms=latency_ms
        )
        
    # ── 단봉 체크 함수 ─────────────────────────
    
    def _hammer(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """ 
        망치형 ─ 하락 추세 후 반전 신호.
        
        조건: 아래꼬리 >= 몸통*2 AND 윗꼬리 <= 몸통*0.3
        해석: 매도세가 강했으나 매수세가 강하게 반격 → 반등 기대.
        신뢰도 0.6: 단봉 패턴은 확인이 필요하므로 중간 수준.
        """
        bar = bars[-1]
        body = abs(bar.close - bar.open)
        lower_shadow = min(bar.open, bar.close) - bar.low 
        upper_shadow = bar.high - max(bar.open, bar.close)
    
        if body < cfg.sys.zero:                 # 도지봉(몸통 거의 없음) 제외
            return self._NO_SIGNAL
        if lower_shadow >= body * 2 and upper_shadow <= body * 0.3:
            return "BUY", 0.6, "hammer"
    
        return self._NO_SIGNAL
    
    def _shooting_star(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """ 
        유성형(shooting star) ─ 상승 추세 후 반전 신호
        
        조건: 윗꼬리 >= 몸통 * 2 and 아래꼬리 <= 몸통 * 0.3
        해석: 매수세가 강했으나 매도세가 강하게 반격 → 하락 기대.
        """
        bar = bars[-1]
        body = abs(bar.close - bar.open)
        upper_shadow = bar.high - max(bar.open, bar.close)
        lower_shadow = min(bar.open, bar.close) - bar.low 
        
        if body < cfg.sys.zero:
            return self._NO_SIGNAL
        if upper_shadow >= body * 2 and lower_shadow <= body * 0.3:
            return "SELL", 0.6, "shooting_star"
        return self._NO_SIGNAL
    
    def _bullish_engulfing(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """ 
        강세 장악형(Bullish Engulfing) ─ 강력한 반전 신호.
        
        조건:
          - 전봉: 음봉 (close < open)
          - 현봉: 양봉 (close > open)
          - 현봉 시가 < 전봉 종가 (갭 다운 시작 → 매도 압력이 컸으나..)
          - 현봉 종가 > 전봉 시가 (이전 몸통 전체를 포괄 → 강한 매수세 예상)
          ∴ 신뢰도 0.7: 이중봉 확인으로 단봉보다 신뢰도가 높음
        """
        if len(bars) < 2:
            return self._NO_SIGNAL
        
        prev, curr = bars[-2], bars[-1]
        if (
            prev.close < prev.open          # 전봉: 음봉
            and curr.close > curr.open      # 현봉: 양봉
            and curr.open < prev.close      # 갭 다운 시가
            and curr.close > prev.open      # 전봉 몸통 완전 포괄
        ):
            return "BUY", 0.7, "bullish_engulfing"
        return self._NO_SIGNAL
    
    def _bearish_engulfing(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """
        약세 장악형(Bearish Engulfing) ─ 상승 후 하락 반전 신호
        
        조건: 전봉 양봉, 현봉 음봉이 전봉 몸통을 완전히 감쌈
        """
        if len(bars) < 2:
            return self._NO_SIGNAL
        
        prev, curr = bars[-2], bars[-1]
        if (
            prev.close > prev.open 
            and curr.close < curr.open 
            and curr.open > prev.close 
            and curr.close < prev.open
        ): 
            return "SELL", 0.7, "bearish_engulfing"
        return self._NO_SIGNAL
    
    def _morning_star(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """ 
        샛별형(Morning Star) ─ 하라 추세에서 반전하는 삼봉 패턴
        
        조건:
          - bar1: 음봉 (하락 지속)
          - bar2: 도지/소형봉 (body2 < b1_body * 0.3) ─ 매수·매도 균형, 방향 탐색
          - bar3: 양봉이 b1 몸통의 중간 이상 회복 ─ 매수세 확인 
          ∴ 신뢰도 0.75: 3봉 확인으로 높은 신뢰도
        """
        if len(bars) < 3:
            return self._NO_SIGNAL
        bar1, bar2, bar3 = bars[-3], bars[-2], bars[-1]
        body2 = abs(bar2.close - bar2.open)
        body1 = abs(bar1.close - bar1.open)
        if (
            bar1.close < bar1.open              # bar1: 음봉
            and body2 < body1 * 0.3             # bar2: 도지 또는 소형봉
            and bar3.close > bar3.open          # bar3: 양봉
            and bar3.close > (bar1.open + bar1.close) / 2   # bar3 종가가 bar1 몸통 중간 이상
        ): 
            return "BUY", 0.75, "morning_star"
        return self._NO_SIGNAL
    
    def _evening_star(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """ 
        저녁별형 (Evening Star) ─ 상승 추세에서 반전하는 삼봉 패턴
        
        morning_star의 반전 버전.
        bar1 양봉 → bar2 도지 → bar3 음봉이 bar1 몸통 중간 이하 하락.
        """
        if len(bars) < 3:
            return self._NO_SIGNAL
        
        bar1, bar2, bar3 = bars[-3], bars[-2], bars[-1]
        body2 = abs(bar2.close - bar2.open)
        body1 = abs(bar1.close - bar1.open)
        if (
            bar1.close > bar1.open 
            and body2 < body1 * 0.3
            and bar3.close < bar3.open 
            and bar3.close < (bar1.open + bar1.close) / 2
        ):
            return "SELL", 0.75, "evening_star"
        return self._NO_SIGNAL
    
    def _box_breakout(self, bars: list[Bar]) -> tuple[Direction, float, str]:
        """ 
        박스권 돌파(Box Breakout) ─ 횡보 구간 이탈 신호.
        
        [알고리즘]
          1. 직전 20봉 (현재봉 제외)의 최고가 = 저항선(resistance)
          2. 직전 20봉의 최저가 = 지지선(support)
          3. 현재 봉 종가가 저항선 * 1.001 초과 → 상단 돌파 (BUY)
          4. 현재 봉 종가가 지지선 * 0.999 미만 → 하단 돌파 (SELL)
          공통. 0.1% 버퍼(1.001, 0.999): 저항선·지지선 바로 위 수준의 잡음을 걸러냄

          ∴ 신뢰도 0.65: 단봉보다 높지만 이중봉보다 낮은 중간 수준
        """
        if len(bars) < 21:
            return self._NO_SIGNAL
        
        box = bars[-21:-1]
        highs = [bar.high for bar in box]
        lows = [bar.low for bar in box]
        resistance = max(highs)
        support = min(lows)
        curr = bars[-1]
        
        if curr.close > resistance * 1.001:
            return "BUY", 0.65, "box_breakout_up"
        if curr.close < support * 0.999:
            return "SELL", 0.65, "box_breakout_down"
        return self._NO_SIGNAL