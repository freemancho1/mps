""" 
HistoricalSimulator ─ 과거 분봉 재생 기반 백테스트 엔진

[전체 파이프라인 한눈에 보기]
  - Bar 리스트 → (BarValidator) → deque 버퍼에 순차 추가
  - 매 봉마다:
    1. 보유 포지션 체크 (StoplossTakeProfitGuard): 청산 조건 만족 시 청산
    2. 룩백 미달 봉은 건너뜀 (지표 계산 불가)
    3. 포지션 있으면 신규 신호 생략 (동시 다중 포지션 없음 ─ Phase 1 단순화)
    4. 피처 추출 + 정규화 → 두 트랙 모델 추론 
    5. 신호 합의 + 지연 필터 + 점수 필터
    6. 수량 계산 → TripleBarrier 기준으로 주문 생성 → 체결
  - 전체 거래 기록 → PerformanceEvaluator → PerformanceReport 반환
  
[단순화 사항 (Phase-1 기준)]
  - 단일 포지션: open_order 변수 하나로 관리 (동시 다중 포지션 없음)
  - 항상 시장가 즉시 체결
  - 공매도는 코드 상 지원하나 Phase-1 신호 필터링으로 사실상 미 발생
"""
from __future__ import annotations 

from collections import deque 

from mps.sys import cfg, msg, MPF_STYLE
from mps.sys.core.types import Bar 
from mps.pipline.evaluator import PerformanceReport
from mps.pipline.features.validator import BarValidator


class HistoricalSimulator: 
    def __init__(
        self,
        capital: float = cfg.run.capital,
        lookback_minutes: int = cfg.sys.lookback_minutes
    ) -> None:
        self._capital = capital 
        self._lookback_minutes = lookback_minutes
        print(msg.hs.init(self))
        
        self._validator = BarValidator()
        
    def run(self, bars: list[Bar]) -> None:
        print(msg.hs.run_info(bars))
        bars = self._validator.filter(bars)
        # TODO: 3. 여기서 부터 진행
        
        return
        
