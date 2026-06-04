""" 
LatencyMonitor ─ 컴포넌트별 지연시간 측정 및 병목 식별.

[역할]
  HistoricalSimulator의 주요 단계(feature·numeric/pattern)를 블록으로 감싸
  실행 시간을 기록하고, 백테스트 완료 후 시뮬레이션 결과를 검증하기 위한 자료를 남김
  
[작동 방식: 컨텍스트 매니저 방식]
  - with monitor.measure("component_name"):
        # 측정할 코드 블록
    → 블록 진입·종료 시 자동으로 시간 측정 (try-finally 방식)
"""
from __future__ import annotations 

import time 
import numpy as np
from typing import Any, Generator 
from collections import defaultdict 
from contextlib import contextmanager

from mps.config import cfg, msg


class LatencyMonitor:
    def __init__(self) -> None:
        # 컴포넌트명 → 지연시간(ms) 리스트
        # defaultdict(type)은 해당 변수(self._records)에 없는 key로 호출하면
        #   {}로 정의된 딕셔너리는 KeyError가 발생하는데, 
        #   defaultdict으로 정의한 변수는 키가 없으면
        #   해당 키를 생성하고 해당 타입의 기본 데이터형을 만듬  
        self._records: dict[str, list[float]] = defaultdict(list)
        
    @contextmanager
    def measure(self, component: str) -> Generator[Any, Any, Any]:
        """ 
        컨텍스트 매니저로 코드 블록 실행 시간 측정
        
        my_latency_checker = LatencyMonitor()        
        with my_latency_checker.measure("feature"):
            raw = ....
        → with가 종료되면 컴포넌트명 "feature"란 키에 실행 시간(ms)이 추가됨
        """
        start_time = time.perf_counter()
        yield   # 이 부분에서 제어권을 with 블럭으로 넘김
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # print(msg.trade.o.latency(elapsed_ms))    # Latency MS: 3.0921380002837395
        self._records[component].append(elapsed_ms)
        
    def record(self, component: str, latency_ms: float) -> None:
        """ 컨텍스트 매니저 대신 직접 값을 기록하고자 할 때 외부에서 직접 호출해 사용 """
        self._records[component].append(latency_ms)
        
    def summary(self) -> dict[str, dict]:
        """ 각 컴포넌트의 count·mean·p95·max 지연시간 반환. """
        result: dict[str, dict] = {}
        for component, values in self._records.items():
            arr = np.array(values)
            result[component] = {
                cfg.key.count: len(arr),
                cfg.key.mean_ms: round(float(arr.mean()), 2),
                cfg.key.p95_ms: round(float(np.percentile(arr, 95)), 2),
                cfg.key.max_ms: round(float(arr.max()), 2)
            }
        return result 
        
