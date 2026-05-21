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

from typing import Optional 

from mps.sys import cfg, msg
from mps.sys.core.types import Bar 


class WalkForwardValidator:
    """ 
    Walk-Forward: 버퍼 구간을 슬라이딩하면서 복수 구간 검증.
    
    각 윈도우마다 독립 HistoricalSimulator인스턴스를 생성하므로 상태 오업 없이
    격리된 평가가 보장됨.
    """
    def __init__(
        self, 
        buffer_days: int = cfg.sys.buffer_days,
        test_days: Optional[int] = None,
        capital: Optional[float] = None
    ) -> None:
        self._buffer_days = buffer_days 
        self._test_days = test_days 
        self._capital = capital 
        