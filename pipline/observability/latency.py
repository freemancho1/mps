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
