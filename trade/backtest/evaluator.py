""" 
PerformanceEvaluator ─ 백테스트 성과 지표 계산.

[계산하는 지표]
  - total_trades            : 완결 거래 수 (진입과 청산 쌍)
  - win_rate                : 수익 거래 비율
  - profit_factor           : 총 수익 / 총 손실 (1 이상이면 수익 > 손실)
  - max_drawdown            : 자산 고점 대비 최대 낙폭 (음수 ─ 클수록 나쁨)
  - sharpe_ratio            : 연환산 샤프 비율 (위험 조정 수익률)
  - total_return_pct        : 전체 기간 총 수익률 (거래당 수익률 합산)
  - avg_return_per_trade    : 거래당 평균 수익률
  - total_cost              : 전체 기간 총 거래 비용
  
[샤프 비율 계산 방식]
  - 거래당 수익률 배열에서 mean/std를 구현한 후 연환산
  - annualization factor(연율화(연환산) 계수) = sqrt(242 * 390)
    · 242: 연간 거래일 수 (2025년 기준)
    · 390: 하루 분봉 수(09:00 ~ 15:30)

  ⇒ 분봉 단위 전략을 연간 수익률 기준으로 비교 가능하게 만듦.
     ─ 거래 빈도가 낮으면 샤프 비율이 극단값을 가질 수 있으니 주의 필요.
     
[최대 낙폭 계산]
  - 누적 수익률 배열 → 누적 고점 → (현재 - 고점) / 고점
    ─ 고점 대비 최악의 순간에 얼마나 떨어졌는가 확인.
"""
from __future__ import annotations 

import numpy as np 

from mps.config import cfg, msg 
from mps.core.types import TradeRecord, PerformanceReport


class PerformanceEvaluator:
    def evaluate(
        self, 
        trades: list[TradeRecord], 
        init_capital: float, 
    ) -> PerformanceReport:
        """ 
        TradeRecord 리스트를 받아 PerformanceReport 반환
        - 거래 정보가 없으면 모든 지표가 0인 빈 보고서 반환.
        """
        # TODO X: 추후 PerformanceEvaluator가 필요할 때 구현
        pass