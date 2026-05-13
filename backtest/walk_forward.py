""" 
WalkForwardValidator: 거래일 단위 슬라이딩 윈도우 검증(순차적)

[Walk-Forward 검증이 필요한 이유]
· Walk-Forward는 시장의 비정상성(regime change)을 고려하여,
  여러 시기에 걸쳐 전략이 일관되게 작동하는지 확인하기 위해 필요함

[슬라이딩 윈도우 방식]
· 버퍼 2거래일 + 테스트 10거래일을 하나의 윈도우로 설정.
· 윈도우를 test_days(10 거래일)씩 앞으로 슬라이딩.
  (step = test_days 이므로 각 거래일이 테스트셋에 정확히 한 번만 등장)

· 버퍼 크기 근거:
  * lookback_minutes(120봉) / 390봉 = 0.31일 → 올림 1일 + 여유 1일 = 2일.
  * 1년(252 거래일) 기준 윈도우 수: (252 - 12(=버퍼+테스트)) / 10(=테스트) = 24개.
  * 예시: 총 120 거래일, 버퍼=2, 테스트=10
    - [0~1]=버퍼1, [2~11]=테스트1 → 레포트1
    - [10~11]=버퍼2, [12~21]=테스트2 → 레포트2  (테스트1과 테스트2에 곂치는 값 없음)
    - ...
  * 결과 해석: '레포트'들의 평균 승률·샤프 비율이 일관되면 전략이 안정적이라는 의미.
               특정 구간에서만 좋으면 시장 국면(regime)에 의존하는 전략일 가능성이 높음
"""
from __future__ import annotations

import math 

from mps.sys.config import settings
from mps.data.types import Bar 
# TODO 2: backtest/simulator.py 작업 후 여기서부터 수행