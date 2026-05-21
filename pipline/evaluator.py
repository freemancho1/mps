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
  - total_cost              : 전체 기간 총  거래 비용
  
[샤프 비율 계산 방식]
  - 거래당 수익률 배열에서 mean/std를 구현 후 연환산
  - anjualization factor = sqrt(252 * 390)
    · 252: 연간 거래일 수
    · 390: 하루 분봉 수 (09:00 ~ 15:30)
  = 분봉 단위 전략을 연간 수익률 기준으로 비교 가능하게 만듦.
  * 주의: 거래 빈도가 낮으면 샤프 비율이 극단값을 가질 수 있음.
  
[최대 낙폭 계산]
  - 누적 수익률 배열 → 누적 고점 → (현재-고점)/고점
    ⇒ 고점 대비 최악의 순간에 얼마나 떨어졌는가?
"""
from __future__ import annotations 

import numpy as np 
from dataclasses import dataclass 


@dataclass 
class PerformanceReport:
    """ 백테스트 성과 요약 보고서 """
    total_trades: int 
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_return_pct: float
    avg_return_per_trade_pct: float 
    total_cost: float 

    def __str__(self) -> str:
        return (
            f"총 거래: {self.total_trades}건 | "
            f"승률: {self.win_rate:.1%} | "
            f"수익 인수: {self.profit_factor:.2f} | "
            f"최대 낙폭: {self.max_drawdown:.1%} | "
            f"샤프: {self.sharpe_ratio:.2f} | "
            f"총 수익률: {self.total_return_pct:.2%} | "
            f"거래당 수익률: {self.avg_return_per_trade_pct:.4%} | "
            f"총 비용: {self.total_cost:,.0f}원"
        )