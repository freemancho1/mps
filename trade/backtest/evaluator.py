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
        if not trades:
            return PerformanceReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # ── 거래당 순 수익률 계산 ──────────
        results = []
        for trade in trades:
            if trade.direction == cfg.key.BUY:
                # BUY: (청산가 - 진입가) / 진입가
                result = (trade.exit_price - trade.entry_price) / trade.entry_price
            else:
                # SELL: (진입가 - 청산가) /진입가
                result = (trade.entry_price - trade.exit_price) / trade.entry_price
            # 비용 차감: 비용 / 진입가 → 비율로 환산
            # 1e-8: entry_price * quantity = 0인 엣지 케이스 방지
            result -= trade.cost / (trade.entry_price * trade.quantity + cfg.run.zero)
            results.append(result)

        result_arr = np.array(results)
        wins_arr = result_arr[result_arr > 0]       # 수익 거래 수익률 배열
        losses_arr = result_arr[result_arr <= 0]    # 손실 거래 수익률 배열 (본전 포함)

        # ── 기본 통계 ───────────────
        win_rate = len(wins_arr) / len(result_arr)
        # profit_factor: 총 수익 / 총 손실 절대값
        # 아래 설명과 도출 절차에 대해서는 나중에 설명을 들어봐야겠음.
        # ─ losses가 없으면 inf (모든 거래가 수익?) → 너무 좋으면 오히려 의심 필요
        # ─ losses_arr는 np.ndarray이며, 원소가 2개 이상이면 value error.
        #    → 손실 거래가 하나라도 있으면 ProfitFactor 계산, 아니면 inf
        profit_factor = (
            float(wins_arr.sum()) / float(-losses_arr.sum() + cfg.run.zero)
                if losses_arr.size else np.inf
        )
        result_total = float(result_arr.sum())  # 단순 수익률 합 (복리 아니며 대략적인 지표임)
        result_avg = float(result_arr.mean())   # 거래당 평균 순 수익률

        # ── 샤프 비율 (역환산) ──────────
        # 거래당 수익률 계산 ─ 하루 390개 분봉 * 연 242 거래일(25년 기준)
        sharpe = float(
            result_arr.mean() / (result_arr.std() + cfg.run.zero)
            * np.sqrt(cfg.run.days_per_year * cfg.run.minutes_per_day)  # 242봉 * 390봉
        )

        # ── 최대 낙폭 (누적 수익률 계산) ──────
        # 1. 누적 수익률 계산: 각 거래 후 자산 변화를 곱으로 표현
        cumulative_return = np.cumprod(1 + result_arr)
        # 2. 누적 고점: 각 시점까지의 최대값
        peak = np.maximum.accumulate(cumulative_return)
        # 3. 낙폭 ─ (현재 - 고점) / 고점 (음수이며 클수록 나쁨)
        drawdown = (cumulative_return - peak) / (peak + cfg.run.zero)
        # 4. 최대 낙폭
        max_drawdown = float(drawdown.min())

        # ── 전체 사용된 비용(수수료 함) ──────
        total_cost = sum(trade.cost for trade in trades)

        return PerformanceReport(
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            total_return_pct=result_total,
            avg_return_per_trade_pct=result_avg,
            total_cost=total_cost
        )
