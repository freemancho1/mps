""" 
Phase-1 단계 패턴 모델 ─ RuleBasedPatternEngine: 캔들스틱·차트 패턴 룰 기반 판정

[7가지 패턴 ─ 우선 순위순]
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
  - Phase-2에서 CNN 기반 차트 이미지 인식으로 일부 대체
  - source 필드를 "RULE"에서 "CNN"으로 변경하면 상위 레이어 변경 없이 통합 가능.
"""
from __future__ import annotations 

import time 

from mps.config import cfg, msg 
from mps.core.types import Bar, SignalDirection, PatternInput, PatternSignal


class RuleBasedPatternEngine:
    
    def _detect(self, bars: list[Bar]) -> tuple[SignalDirection, float, str]:
        """ 
        아래 우선순위 순(check_func)으로 패턴을 검사하고 첫 번째 매칭을 그대로 반환.
        - bars는 정규화하지 않은 원본 데이터 사용 (비율 계산에 필요함)
        """
        if len(bars) < 5:
            return cfg.trade.signal.no_signal_pattern
        
        # v3부터는 "롱 온리" 형태로 "매수 패턴만 진입 신호로 사용".
        # → 하락 패턴인 유성형·약세장악·저녁별·하단돌파 형식은 매수 진입 근거가 아니므로 제외
        # TODO 0615-1633 하단 함수 정의 후 수행
        # check_func = [
        #     self._hammer,
        #     self._bullish_engulfing,
        #     self._morning_star,
        #     self._box_breakout
        # ]
        
        
    # ── 체크 함수 정의 ───────────────────
    
    def _hammer(self, bars: list[Bar]) -> tuple[SignalDirection, float, str]:
        """ 
        망치형 신호 ─ 하락 추세 후 반전 신호
        
        조건: 아래꼬리 >= 몸통 * 2 and 윗꼬리 <= 몸통 * 0.3
        해석: 매도세가 강했으나 메수세가 강하게 반격 → 가격 반등 기대
        신뢰도: 0.6 ─ 단봉 패턴은 확인이 필요하므로 최하 수준 신뢰도.
        """
        curr = bars[-1]
        body = abs(curr.close - curr.open)
        lower_shadow = min(curr.open, curr.close) - curr.low 
        upper_shadow = curr.high - max(curr.open, curr.close)
        
        if body < cfg.sys.zero:         # 도지봉(몸통이 거의 없는 봉) 제외
            return cfg.trade.signal.no_signal_pattern
        
        if lower_shadow >= body * 2 and upper_shadow <= body * 0.3:
            return cfg.str.buy, cfg.trade.pc.single_confidence, cfg.str.hammer
        return cfg.trade.signal.no_signal_pattern
    
    def _bullish_engulfing(self, bars: list[Bar]) -> tuple[SignalDirection, float, str]:
        """ 강세 장악형(불장, 이중봉) ─ 강력한 반전 신호 기대 """
        if len(bars) < 2:
            return cfg.trade.signal.no_signal_pattern
        
        prev, curr = bars[-2], bars[-1]
        if (
            prev.close < prev.open          # 전봉 음봉
            and curr.close > curr.open      # 현봉 양봉
            and curr.open < prev.close      # 갭 다운
            and curr.close > prev.open      # 전봉 몸통을 완전 포함
        ):
            return cfg.str.buy, cfg.trade.pc.double_confidence, cfg.str.bullish_engulfing
        return cfg.trade.signal.no_signal_pattern
    
    def _morning_star(self, bars: list[Bar]) -> tuple[SignalDirection, float, str]:
        """ 샛별형(삼봉) ─ 하락 추세에서 반전하는 삼봉 패턴 """
        if len(bars) < 3:
            return cfg.trade.signal.no_signal_pattern
        
        bar1, bar2, bar3 = bars[-3], bars[-2], bars[-1]
        body2 = abs(bar2.close - bar2.open)
        body1 = abs(bar1.close - bar1.open)
        if (
            bar1.close < bar1.open          # bar1 음봉
            and body2 < body1 * 0.3         # bar2가 도지 또는 소형봉
            and bar3.close > bar3.open      # bar3 양봉
            and bar3.close > (bar1.open + bar1.close) / 2   # bar3 종가가 bar1 몸통 중간 이상
        ):
            return cfg.str.buy, cfg.trade.pc.triple_confidence, cfg.str.morning_star
        return cfg.trade.signal.no_signal_pattern
    
    def _box_breakout(self, bars: list[Bar]) -> tuple[SignalDirection, float, str]:
        """ 
        박스권 돌파(Box Breakout) ─ 횡보 구간에서 이탈 신호
        
        [알고리즘]
          1. 직전 20봉(현재봉 제외)의 최고가 = 저항선(resistance)
          2. 직전 20봉의 최저가 = 지지선(support)
          3. 현재봉 종가가 저항선 * 1.001 초과 → 상단 돌파(BUY)
          4. 현재봉 종가가 지지선 * 0.999 미만 → 하단 돌파(SELL)
          
        [신뢰도] 
          0.65로 단봉(0.6)보다는 높지만 이/삼중봉 보다 낮음
          ─ 이 조건은 아주 좋은 조건이므로 다음에는 우선순위를 1순위로 하는것 검토
        """
        if len(bars) < cfg.trade.pc.chart_min_size:
            return cfg.trade.signal.no_signal_pattern
        
        box = bars[-cfg.trade.pc.chart_min_size:-1]
        highs = [bar.high for bar in box]
        lows = [bar.low for bar in box]
        resistance = max(highs)
        support = min(lows)
        curr = bars[-1]
        
        if curr.close > resistance * 1.001:
            return cfg.str.buy, cfg.trade.pc.chart_confidence, cfg.str.box_breakout_up
        # 롱 오니: 하단 돌파(box_breakout_down)는 매수 진입 신호가 아니므로 사용하지 않음.
        return cfg.trade.signal.no_signal_pattern
    
    
    # ── 체크 함수 정의 (롱 온리 형태에서는 사용하지 않음) ───
    
    # "롱 온리" 타입에서는 매수(BUY)와 관망(HOLD)만 존재하기 때문에
    # SignalDirection에는 "SELL"이 존재하지 않아,
    # 아래 코드에서는 Pylance 오류가 발생하는데, 주석 처리하니 무시.
    
    # def _shooting_star(self, bars: list[Bar]) -> tuple[SignalDirection, float, str]:
    #     """ 
    #     유성형 ─ 상승 추세 후 반전 신호 (하락장 지표로 진입 대상 아님)
        
    #     조건: 윗꼬리 >= 몸통 * 2 and 아래꼬리 <= 몸통 * 0.3
    #     해석: 매수세가 강했으나 매도세가 강하게 반격 → 가격 하락 예상
    #     """
    #     curr = bars[-1]
    #     body = abs(curr.close - curr.open)
    #     upper_shadow = curr.high - max(curr.open, curr.close)
    #     lower_shadow = min(curr.open, curr.close) - curr.low 
        
    #     if body < cfg.sys.zero:         # 도지봉(몸통이 거의 없는 봉) 제외
    #         return cfg.trade.signal.no_signal_pattern
        
    #     if upper_shadow >= body * 2 and lower_shadow <= body * 0.3:
    #         return cfg.str.sell, cfg.trade.pc.single_confidence, cfg.str.shooting_star
    #     return cfg.trade.signal.no_signal_pattern
        
    # def _bearish_engulfing(self, bars: list[Bar]) -> tuple[SignalDirection, float, str]:
    #     """ 약세 장악형(곰장) ─ 상승 후 하락 반전 신호 """
    #     if len(bars) < 2:
    #         return cfg.trade.signal.no_signal_pattern
        
    #     prev, curr = bars[-2], bars[-1]
    #     if (
    #         prev.close > prev.open          # 전봉 양봉
    #         and curr.close < curr.open      # 현봉 음봉
    #         and curr.open > prev.close 
    #         and curr.close < prev.open
    #     ):
    #         return cfg.str.sell, cfg.trade.pc.double_confidence, cfg.str.bearish_engulfing
    #     return cfg.trade.signal.no_signal_pattern
    
    # def _evening_star(self, bars: list[Bar]) -> tuple[Direction, float, str]:
    #     """ 저녁별형 ─ 상승 추세에서 반전하는 삼봉 패턴 (샛별형 반대) """
    #     # 아래 코드는 수정되지 않았음.. 필요시(그럴일 없지만) 변경해서 작성할 것
    #     if len(bars) < 3:
    #         return cfg.run.no_signal_pattern
        
    #     bar1, bar2, bar3 = bars[-3], bars[-2], bars[-1]
    #     body2 = abs(bar2.close - bar2.open)
    #     body1 = abs(bar1.close - bar1.open)
    #     if (
    #         bar1.close > bar1.open 
    #         and body2 < body1 * 0.3
    #         and bar3.close < bar3.open 
    #         and bar3.close < (bar1.open + bar1.close) / 2
    #     ):
    #         return cfg.key.SELL, cfg.run.pc.triple_peak, cfg.str.evening_star
    #     return cfg.run.no_signal_pattern