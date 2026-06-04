""" 
WalkForwardValidator ─ 거래일 단위 슬라이딩 윈도우 검증 (순차적)

[Walk-Forward 검증이 필요한 이유]
  - Walk forward는 시장의 비정상성(regime change)을 고려하여,
    여러 시기에 걸처 전략이 일관되게 작동하는지 확인하기 위해 필요함.
    
[슬라이딩 윈도우 방식]
  - 버퍼 2거래일 + 테스트 10(임의 설정=2주) 거래일을 하나의 윈도우로 설정.
  - 윈도우를 테스트 일자(10 거래일, 테스트 실행 시 입력값으로 받음)씩 앞으로 슬라이딩.
    (step = test_days 이므로 각 거래일이 테스트셋에 정확히 한 번만 등장)
  - 버퍼 크기 근거:
    · lookback_minutes(120봉) / 일(390봉) = 0.31일 ⇒ 올림(1일) + 여유(1일) = 2일
    · 1년(242일=2025년도) 기준 윈도우 수: (242 - 12(버퍼+테스트 일)) / 10(테스트 일) = 23개.
    · '레포트'들의 평균 승률·샤프 비율이 일관되면 전략이 안정적이라는 의미.
      특정 구간에서만 좋으면 시장 국면(regime)에 의존하는 전략일 가능성이 있음.
"""
from __future__ import annotations

import math 

from mps.config import cfg, msg 
from mps.core.types import Bar, PerformanceReport
from .simulator import HistoricalSimulator


class WalkForwardValidator:
    """ 
    Walk-Forward: 버퍼 구간을 슬라이딩하면서 복수 구간 검증.
    
    각 윈도우마다 독립 HistoricalSimulator 인스턴스를 생성하므로,
    상태 오염없이 격리된 평가가 보장됨.
    """
    def __init__(
        self, 
        buffer_days: int = cfg.run.buffer_days,
        test_days: int = cfg.run.test_days,
        capital: float = cfg.run.init_capital
    ) -> None:
        self._buffer_days = buffer_days
        self._test_days = test_days 
        self._capital = capital 
        print(msg.trade.bt.wf_info(buffer_days, test_days, capital))
        
    def run(self, bars: list[Bar]) -> list[PerformanceReport]:
        """ 
        전체 bars에 걸쳐 슬라이딩 윈도우를 적용하고 각 구간의 성과 보고서 반환.
        
        [윈도우 진행 방식]
          - 윈도우 크기 = buffer_days + test_days
          - 시작 위치(start_idx)를 test_days씩 증가
          
        [ValueError 처리]
          - 데이터 부족 구간(lookback 미달)은 건너 뜀.
            ─ 일반적으로 첫 윈도우는 데이터가 부족해 skip함. 
        """
        all_days = sorted({bar.timestamp.date() for bar in bars})
        num = len(all_days)
        window_size = self._buffer_days + self._test_days
        
        reports:list[PerformanceReport] = []
        
        for start_idx in range(0, num - window_size, self._test_days):
            buffer_end = start_idx + self._buffer_days
            test_end = buffer_end + self._test_days
            
            if test_end > num:
                break   # 마지막 불완전한 윈도우는 테스트 대상이 아님.
            
            # 윈도우 전체(버퍼+테스트) Bar 추출
            # 버퍼 구간 = 지표 계산용 룩백
            # 테스트 구간 = 실제 신호 발생 구간 
            window_day_set = set(all_days[start_idx:test_end])
            window_bars = [bar for bar in bars if bar.timestamp.date() in window_day_set]
            
            try:
                simulator = HistoricalSimulator(capital=self._capital)
                report = simulator.run(window_bars)
                reports.append(report)
            except ValueError as ve:
                # 데이터 부족 → 이 윈도우는 skip (정상적으로 발생 가능)
                print(msg.trade.bt.wf_skip_err(str(ve)))
                continue
            
        return reports
               
