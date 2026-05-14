""" 
HistoricalSimulator - 과거 분봉 재생 기반 백테스트 엔진.

[전체 파이프라인]
· Bar 리스트 → (BarValidator) → deque 버퍼에 순차 추가
· 매 봉마다:
  ① 보유 포지션 체크 (StopLossTakeProfitGuard): 청산 조건 만족 시 청산
  ② 룩백 미달 봉은 건너뜀 (지표 계산 불가)
  ③ 포지션 있으면 신규 신호 생략 (동시 다중 포지션 없음 - Phase 1 단순화)
  ④ 피처 추출 + 정규화 → 두 트랙 모델 추론
  ⑤ 신호 합의 + 지연 필터 + 점수 필터
  ⑥ 수량 계산 → Triple Barrier 기준으로 주문 생성 → 체결
· 전체 거래 기록 → PerformanceEvaluator → PerformanceReport 반환

[단순화 사항 (Phase 1)]
· 단일 포지션: open_order 변수 하나로 관리 (동시 다중 포지션 없음)
· 항상 시장가 즉시 체결
· 공매도는 코드 상 지원하나 phase 1 신호 필터링으로 사실상 발생하지 않음
"""
from __future__ import annotations

from collections import deque 
from tqdm import tqdm 

from mps.data.types import Bar, Order
from mps.sys.config import settings
from mps.data.features.validator import BarValidator
from mps.data.features.normalizer import NumericalNormalizer, PatternNormalizer
from mps.models.numerical.extractor import FeatureExtractor
from mps.models.numerical.model import ThresholdModel
from mps.models.pattern.rules import RuleBasedPatternEngine
from mps.signal.aggregator import SignalAggregator
from mps.signal.latency_guard import LatencyGuard
from mps.signal.filter import SignalFilter
# TODO 3: risk 작업 후 처리