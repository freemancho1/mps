""" 
PerformanceEvaluator ─ 백테스트 성과지표 계산

[계산 방식]
  - 거래 기준 지표(승률·profit factor·거래당 평균 수익률)은 그래도 유지하고,
    '순손익(pnl_net, 원화)' 기반으로 통일.
  - 포트폴리오 지표는 에쿼티 곡선(자본 + 누적 순손익)에서 계산:
    · total_return_pct  = Σ pnl_net / init_capital
    · max_drawdown      = 에퀴터 곡선 고점 대비 최대 낙폭
    · sharpe_ratio      = 일별 수익률 mean/std * sqrt(242) ─ 일별 집계
  - '백테스트에서 수익처럼 보였는데 실제로는 아니었다'를 만드는 가장 흔한 원인이
    평가 지표의 부풀림이므로, 이 수정 자체가 수익성 검증의 전제임.
"""
from __future__ import annotations 

import numpy as np 
from collections import defaultdict 

from mps.config import cfg 
from mps.core.types import TradeRecord, PerformanceReport 


class PerformanceEvaluator:
    def evaluate(
        self,
        trades: list[TradeRecord],
        init_capital: float,
    ) -> PerformanceReport:
        # 거래가 없으면 0원 보고서
        if not trades:
            return PerformanceReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # ── 거래 기준 지표
        pnl_arr = np.array([trade.pnl_net for trade in trades], dtype=np.float64)
        # 거래 수익률 = 순손익 / 투입 원금 (진입가 * 수량)
        notional = np.array(
            [trade.entry_price * trade.quantity for trade in trades],
            dtype=np.float64
        )
        ret_arr = pnl_arr / (notional + cfg.sys.zero)

        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]  # 본전 포함

        win_rate = len(wins) / len(pnl_arr)
        # profit_factor = 총 이익 / |총 손실| ─ 손실 거래가 없으면 inf (의심 필요)
        profit_factor = (
            float(wins.sum()) / float(-losses.sum() + cfg.sys.zero)
                if losses.size else float("inf")
        )
        avg_return_per_trade = float(ret_arr.mean())

        # ── 포트폴리오 기준 지표
        # 1. 총 수익률: 초기 자본 대비 순손익 합
        total_pnl = float(pnl_arr.sum())
        total_return_pct = total_pnl / (init_capital + cfg.sys.zero)

        # 2. 최대 낙폭: 청산 시각 순으로 에쿼티 곡선 구성 후 고점 대비 낙폭
        order = sorted(range(len(trades)), key=lambda idx: trades[idx].exit_time)
        equity = init_capital + np.cumsum(pnl_arr[order])
        equity = np.concatenate([[init_capital], equity])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + cfg.sys.zero)
        max_drawdown = float(drawdown.min())

        # 3. 샤프: 일별 순손익 → 일별 수익률 → 연환산(sqrt(242))
        daily_pnl: dict = defaultdict(float)
        for trade in trades:
            daily_pnl[trade.exit_time.date()] += trade.pnl_net 

        daily_ret = np.array(
            [v / init_capital for _, v in sorted(daily_pnl.items())],
            dtype=np.float64
        )
        if daily_ret.size >= 2:
            sharpe = float(
                daily_ret.mean() / (daily_ret.std() + cfg.sys.zero)
                * np.sqrt(cfg.market.days_per_year)
            )
        else:
            sharpe = 0.0    # 표본이 1일 이하면 통계적으로 무의미

        total_cost = float(sum(trade.cost for trade in trades))

        return PerformanceReport(
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            total_return_pct=total_return_pct,
            avg_return_per_trade_pct=avg_return_per_trade,
            total_cost=total_cost
        )