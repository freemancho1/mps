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

from typing import Optional 
from collections import deque 

from mps.sys import cfg, msg, MPF_STYLE
from mps.sys.core.types import Bar, Order, NumericalInput, PatternInput
from mps.pipline.evaluator import PerformanceReport, TradeRecord
from mps.pipline.features.validator import BarValidator
from mps.pipline.features.normalizer import NumericalNormalizer, PatternNormalizer
from mps.pipline.models.numerical.extractor import FeatureExtractor
from mps.pipline.observability.latency import LatencyMonitor


class HistoricalSimulator: 
    def __init__(
        self,
        capital: float = cfg.run.capital,                   # 10,000,000.0원
        lookback_minutes: int = cfg.sys.lookback_minutes    # 120
    ) -> None:
        self._capital = capital 
        self._lookback_minutes = lookback_minutes
        # print(msg.hs.init(self))
        
        self._validator = BarValidator()
        self._extractor = FeatureExtractor()
        self._numeric_normalizer = NumericalNormalizer()
        self._pattern_normalizer = PatternNormalizer()
        self._latency = LatencyMonitor()
        
    def run(self, bars: list[Bar]) -> None:
        print(msg.hs.run_info(bars))
        # is_complete=False 봉이 섞여 있으면 look-ahead bias 발생 위험
        bars = self._validator.filter(bars)

        # 룩백 + 1 이상 없으면 의미 있는 백테스트 불가
        if len(bars) < self._lookback_minutes + 1:
            raise ValueError(msg.hs.lookback_under_err(bars, self._lookback_minutes+1))
        
        # 상태변수 초기화
        # maxlen = lookback + 50: 가장 오래된 봉이 자동 삭제 → 메모리 효율
        # +50은 기술 지표 초기화 구간(NaN봉)을 여유롭게 포함하기 위함
        buffer: deque[Bar] = deque(maxlen=self._lookback_minutes + 50)  # 120+50 = 170
        # print(msg.hs.size_check(bars, buffer))
        trades: list[TradeRecord] = []      # 완결된 거래에 대한 기록
        cash = self._capital                # 현재 사용 가능한 현금 (총 자산)
        open_order: Optional[Order] = None  # 현재 보유 중인 포지션 (None = 미보유)
        
        # ── 메인 루프: 봉 하나씩 생성 ─────────────────────
        for bar in bars:
            buffer.append(bar)

            # ── 1. 현재 보유중인 포지션이 있으면 청산 체크 ───────────
            # open_order가 있으면 현재 봉 종가로 TP·SL·만료 조건 확인
            if open_order is not None:
                # TODO: 4 여기 작성해야 함
                pass 

            # ── 2. 룩백 미달 구간은 신호 생성 생략 ───────────────
            # buffer에 lookback 봉 이상 쌓이기 전까지는 지표 계산이 의미 없음
            if len(buffer) < self._lookback_minutes:
                continue 

            # ── 3. 현재 미처리 포지션이 있으면 신규 구매 생략(한번에 하나의 거래만) ──
            if open_order is not None:
                continue

            # deque 슬라이싱을 위해 list로 변환
            buffer_list = list(buffer)

            # ── 4. 피처 추출 및 정규화 ───────────────────────
            with self._latency.measure("feature"):
                raw = self._extractor.extract(buffer_list)
                numeric_input = self._numeric_normalizer.transform(buffer_list, raw)
                pattern_input = self._pattern_normalizer.transform(buffer_list)

        
        return
        
