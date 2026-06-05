""" 
모델 팩토리 ─ cfg.run.phase에 따라 트랙별 모델을 선택

[Phase 전환]
  - phase == 1: [Num] ThresholdModel(룰), [Pat] RuleBasedPatternEngine(룰)
  - phase >= 2: [Num] LSTMModel(학습), [Pat] CNN1DPatternModel(학습)
  
[안전 풀백]
  - phase >= 2이지만 학습 가중치가 없으면 해당 트랙은 Phase-1로 폴백함.
    (학습을 하지 않는 상태에서 무작위 초기 모델로 매매하는 사고 방지)
  - 풀백 시 경고를 출력해 관측 가능하게 구현
  
[교체 비용 = 0]
  - 반환 타입이 NumericModelPort·PatternModelPort를 만족하므로,
    시뮬레이터는 어떤 모델이 반환되든 동일하게 .run()만 호출하면 됨.
"""
from __future__ import annotations

# TODO 2: phase-1 단계 모델 구현 후 (ThresholdModel, RuleBasedPatternEngine)