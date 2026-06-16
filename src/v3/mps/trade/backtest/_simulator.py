""" 
HistoricalSimulator ─ 과거 분봉 재생 기반 백테스트 엔진

[전체 파이프라인]
  - Bar 리스트 → (BarValidator 통과) → 전체 deque 버퍼에 순차 추가
  - 매 봉마다:
    1. 보유 포지션 체크 (StopLossTakeProfitGuard): 청산 조건 만족 시 청사
    2. 룩백 미달 봉은 건너뜀 (지표 계산 불가)
    3. 포지션 있으면 신규 신호 생략 (동시 다중 포지션 없음 ─ Phase-1 단순화)
    4. 피처 추출 + 정규화 → 두 트랙 모델 추론
    5. 신호 합의 + 지연 필터 + 점수 필터
    6. 수량 계산 → TripleBarrier 기준으로 주문 생성 → 체결
  - 전체 거래 기록 → PerformanceEvaluator → PerformanceReport 반환
  
[로직 단순화]
  - 단일 포지션: open_order 변수 하나로 관리 (동시에 다중 포지션 없음)
  - 항상 시장가 체결
  - 롱 온리: 모든 포지션은 매수 진입 → 매도 청사 (공매도 없음)
"""
from __future__ import annotations 

from typing import cast, Optional 
from datetime import datetime 
from collections import deque 
from dataclasses import replace 

from mps.config import cfg, msg 
from mps.core.types import Bar, Order, TradeRecord, PerformanceReport
from mps.core.types import ExitReason
# TODO 0616-1546
