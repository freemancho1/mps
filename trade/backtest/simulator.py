""" 
HistoricalSimulator ─ 과거 분봉 재생 기반 백테스트 엔진

[전체 파이프라인 한눈에 보기]
  - Bar 리스트 → (BarValidator 통과) → 전체 deque 버퍼에 순차 추가
  - 매 봉마다:
    1. 보유 포지션 체크 (StoplossTakeprofitGuard): 청산 조건 만족 시 청산
    2. 룩백 미달 봉은 건너뜀 (지표 계산 불가)
    3. 포지션 있으면 신규 신호 생략 (동시 다중 포시션 없음 ─ Phase-1 단순화)
    4. 피처 추출 + 정규화 → 두 트랙 모델 추론
    5. 신호 합의 + 지연 필터 + 점수 필터
    6. 수량 계산 → TripleBarrier 기준으로 주문 생성 → 체결
  - 전체 거래 기록 → PerformanceEvaluator → PerformanceReport 반환
  
[Phase-1 단순화 사항]
  - 단일 포지션: open_order 변수 하나로 관리 (동시 다중 포지션 없음)
  - 항상 시장가 체결
  - 공매도는 코드 상 지원하나 Phase-1 신호 필터링으로 사실상 미 발생
"""
from __future__ import annotations 

from typing import Optional 
from collections import deque 
from dataclasses import replace 

from mps.config import cfg, msg 
from mps.core.types import Bar, Order, TradeRecord, PerformanceReport
from mps.core.ports import NumericModelPort, PatternModelPort
from mps.pp.features.validator import BarValidator



class HistoricalSimulator:
    def __init__(
        self,
        capital: float = cfg.run.init_capital,              # 10,000,000.0원
        lookback_minutes: int = cfg.run.lookback_minutes,   # 120
        numeric_model: Optional[NumericModelPort] = None, 
        pattern_model: Optional[PatternModelPort] = None,
    ) -> None:
        self._capital = capital 
        self._lookback_minutes = lookback_minutes
        
        self._validator = BarValidator()
        
    def run(self, bars: list[Bar]) -> PerformanceReport:
        print(msg.trade.bt.sim_info(bars))
        
        # look-ahead bias 발생 위험 제거를 위해 is_complete=False 봉 제거
        bars = self._validator.filter(bars)
        
        # 윈도우 크기가 룩백+1 이상이 안되면 의미있는 백테스트가 불가하기 때문에 스킵
        # 12일(테스트 일자(10일) + 버퍼(2일)) * 390봉 = 4680봉인데, 
        #   이중에 필터링이 아무리 많이 당해봐야 121(lookback_minutes + 1)봉 보다는 많치 않을까? 싶네...
        if len(bars) < self._lookback_minutes + 1:
            raise ValueError(msg.trade.bt.sim_skip_err(bars, self._lookback_minutes + 1))
        
        # 상태변수 초기화
        # maxlen = lookback + 50 → 가장 오래된 봉은 deque 특성상 자동 삭제됨
        #                   + 50 = 기술지표 초기화 구간(NaN봉)을 여유롭게 포함하기 위함
        buffer: deque[Bar] = deque(maxlen=self._lookback_minutes + 50)  # 120 + 50 = 170
        trades: list[TradeRecord] = []      # 완결된 거래에 대한 기록
        cash = self._capital                # 현재(독립된) 시뮬레이터에서 사용할 수 있는 초기 자본금
        open_order: Optional[Order] = None  # 현재 보유 중인 포지션(거래 상태, None = 거래 없음)
        
        # ── 메일 루프: 봉 하나씩 생성 ─────────────────────
        for bar in bars:
            buffer.append(bar)
            
            # ── 1. 현재 보유중인 포지션이 있으면 청산 체크 ───────────
            if open_order is not None:
                # TODO X: 이후 처리
                pass
            
            # ── 2. 룩백 미달 구간은 신호생성 생략 ───────────────
            if len(buffer) < self._lookback_minutes:
                continue
            
            # ── 3. 현재 미처리 포지션이 있으면 신규 구매 생략 ──────────
            if open_order is not None:
                continue 
            
            # ── 4. 신규 구매 절차 ───────────────────────
            buffer_list: list[Bar] = list(buffer)
            
            # 4-1. 피처 추출 및 정규화 --------------
            # TODO 1: latency 작업 + 정규화 작업 종료 후 수행
            # TODO Z: 여기부터 
            
        
        
        return None