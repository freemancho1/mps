""" 
LatencyMonitor ─ 컴포넌트별 지연시간 측정 및 병목 식별.

[역할]
  - HistoricalSimulator의 주요 단계(feature·numerical·pattern)에서 
    with self._latency.measure("feature"): 블록으로 감싸 실행 시간을 기록.

[사용 목적]
  - 백테스트 완료 후 simulator.latency.summary()를 호출하면,
    각 단계의 평균·p95·최대 지연시간을 확인할 수 있음.
  - Phase-2에서 실거래 전환 시 LatencyGuard 임계값(5초)을 실측값으로 보정하는데 활용

[컨텍스트 매니저 방식]
  - with monitor.measure("component_name"):
        # 측정할 코드 블록
    → 블록 진입·종료 시 자동으로 시간 측정 (try-finally 방식)
"""
from __future__ import annotations

import time
from typing import Any, Generator 
import numpy as np
from collections import defaultdict 
from contextlib import contextmanager 


class LatencyMonitor:
    def __init__(self) -> None:
        # 컴포넌트명 → 지연시간(ms) 리스트
        self._records: dict[str, list[float]] = defaultdict(list)
        # print(f"*** self._records: [{self._records}({type(self._records)})]")
        
    @contextmanager
    def measure(self, component: str) -> Generator[Any, Any, Any]:
        """ 
        컨텍스트 매니저로 코드 블록 실행 시간 측정
        
        with self._latency.measure("feature"):
            raw = self._extractor.extract(buf_list)
        → "feature" 키에 실행 시간(ms) 추가
        """
        start_time = time.perf_counter()        # 상대시간 측정용 고해상도(밀리초) 측정
        yield                                   # 여기서 잠깐 제어권을 with 블록으로 넘김
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._records[component].append(elapsed_ms)
        
    def record(self, component: str, latency_ms: float) -> None:
        """ 컨텍스트 매니저 대신 직접 값을 기록할 때 사용 (외부 측정값 입력). """
        self._records[component].append(latency_ms)
        
    def summary(self) -> dict[str, dict]:
        """ 각 컴포넌트의 count·mean·p95·max 지연시간 반환. """
        result: dict[str, dict] = {}
        for component, values in self._records.items():
            arr = np.array(values)
            result[component] = {
                "count": len(arr),
                "mean_ms": round(float(arr.mean()), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "max_ms": round(float(arr.max()), 2),
            }
        
        return result
    