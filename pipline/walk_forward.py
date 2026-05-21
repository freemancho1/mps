""" 
WalkForwardValidator ─ 거래일 단위 슬라이딩 윈도우 검증.

[Walk-Forward 검증이 필요한 이유]
  - Walk-Forward는 시장의 비정상성(regime change)을 고려하여
    여러 시기에 걸처 전력이 일관되게 작동하는지 확인하는 검증
    
[슬라이딩 윈도우 방식]
  - 버퍼 2 거래일 + 테스트 10 거래일을 하나의 윈도우로 설정.
  - 윈도우를 test_days(10거래일)씩 앞으로 슬라이딩하며 반복.
    → 스탭은 test_days이므로 각 거래일이 테스트셋에 정확히 한 번만 등장함.
"""
from __future__ import annotations

from mps.sys import cfg, msg
from mps.sys.core.types import Bar 
from mps.pipline.evaluator import PerformanceReport
from mps.pipline.simulator import HistoricalSimulator


class WalkForwardValidator:
    """ 
    Walk-Forward: 버퍼 구간을 슬라이딩하면서 복수 구간 검증.
    
    각 윈도우마다 독립 HistoricalSimulator인스턴스를 생성하므로 상태 오업 없이
    격리된 평가가 보장됨.
    """
    def __init__(
        self, 
        buffer_days: int = cfg.sys.buffer_days,
        test_days: int = cfg.run.test_days,
        capital: float = cfg.run.capital
    ) -> None:
        self._buffer_days = buffer_days 
        self._test_days = test_days
        self._capital = capital
        print(msg.wfv.init(self))
        
    def run(self, bars: list[Bar]) -> list[PerformanceReport]:
        """ 
        전체 bars에 걸쳐 슬라이딩 원도우를 적용하고 각 구간의 성과 보고서 반환
        
        [윈도우 진행 방식]
          - 윈도우 크기 = buffer_days + test_days 
          - start_idx를 test_days씩 증가
          
        [ValueError 처리]
          - 데이터 부족 구간(lookback 미달)은 건너뜀.
            ⇒ 이 방법은 정상 동작으로 데이터가 부족한 첫 윈도우 skip
        """
        trading_days = sorted({bar.timestamp.date() for bar in bars})
        num_trading_days = len(trading_days)
        window_size = self._buffer_days + self._test_days
        print(msg.wfv.run_info(trading_days, window_size))
        
        reports: list[PerformanceReport] = []
        for start_idx in range(0, num_trading_days - window_size, self._test_days):
            buffer_end = start_idx + self._buffer_days
            test_end = buffer_end + self._test_days
            
            if test_end > num_trading_days: # 마지막 불완전 윈도우 제외
                break   
            
            # 윈도우 전체(버퍼+테스트) Bar 추출
            # 버퍼 구간 = 지표 계산용 룩백, 테스트 구간 = 실제 신호 발생 구간
            window_day_set = set(trading_days[start_idx:test_end])
            window_bars = [bar for bar in bars if bar.timestamp.date() in window_day_set]
            print(msg.wfv.win_bars_info(start_idx, window_bars))
            
            try:
                simulator = HistoricalSimulator(capital=self._capital)
                report = simulator.run(window_bars)
                # TODO: 2. HistoricalSimulator.run() 작업 후
            except ValueError:
                print(msg.wfv.err_win_bars(window_bars))
                continue
        
        return reports
        
        
        