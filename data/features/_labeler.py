""" 
TripleBarrierLabeler ─ Triple Barrier 라벨 생성기 (결정 6)

[역할]
  - 학습 기반 모델(LSTM·1D-CNN)의 정답 라벨 생성
  - 봉 t에서 신호가 났다고 가정하고, '다음 봉(t+1)의 시가'에 진입한 뒤 time_horizon분 동안:
    · 익절선(+take_profit) 먼저 도달 → BUY (매수했어야 할 자리)
    · 손절선(-stop_loss) 먼저 도달 → HOLD (매수 부적합 = 관망)
    · 시간 만료 → HOLD (방향 불명확)
"""
from __future__ import annotations 

import numpy as np 
from typing import Optional 

from mps.core.config import cfg, msg 
from mps.core.types import Bar, SignalDirection
from mps.core.libs import logger


class TripleBarrierLabeler:
    def __init__(
        self,
        take_profit: Optional[float] = None, 
        stop_loss: Optional[float] = None,
        time_horizon: Optional[int] = None,
    ) -> None:
        self._take_profit: float = cfg.barrier.take_profit \
            if take_profit is None else take_profit 
        self._stop_loss: float = cfg.barrier.stop_loss \
            if stop_loss is None else stop_loss 
        self.time_horizon: int = cfg.barrier.time_horizon \
            if time_horizon is None else time_horizon
            
    def label(self, bars: list[Bar]) -> np.ndarray:
        """ 
        각 봉 t의 Triple Barrier 라벨 (클래스 인덱스) 반환.
        
        진입가 = Open[t+1]
          - 장벽 스캔 구간 = 봉 t+1 ~ t+horizon
            (진입 봉 포함: 시가 체결 후 그 봉의 고·저가가 장벽에 닿을 수 있음)
        
        반환값: shape [len(bars)] int 64
          - 마지막 봉(다음 봉 없음)과 horizon 미확정 구간은 HOLD
            → 학습 시 Dataset에서 마지막 horizon개 시점을 제외해 라벨 누수 방지
        """
        bar_count = len(bars)
        
        open_arr = np.fromiter((b.open for b in bars), dtype=np.float64, count=bar_count)
        high_arr = np.fromiter((b.high for b in bars), dtype=np.float64, count=bar_count)
        low_arr  = np.fromiter((b.low  for b in bars), dtype=np.float64, count=bar_count)
        
        label_arr = np.full(bar_count, cfg.data.dir2idx[cfg.str.hold], dtype=np.int64)
        if bar_count < 2:
            return label_arr 
        
        # 시점 t(0..bar_count-2)의 진입가 = 다음 봉 시가
        entry = open_arr[1:]
        take_profit_line = entry * (1.0 + self._take_profit)
        stop_loss_line = entry * (1.0 - self._stop_loss)
        
        # 정수형(np.int64) 값 중에 가장 큰 값 저장
        big = np.iinfo(np.int64).max 
        # 익절·손절 최초 도달 오프셋으로 초기값은 동일 값.
        first_take_profit = np.full(bar_count - 1, big, dtype=np.int64)
        first_stop_loss = np.full(bar_count - 1, big, dtype=np.int64)
        
        # k=진입봉(t+1) 기준 오프셋. k=0이 진입 봉 자신
        for k in range(self.time_horizon):
            n_valid = bar_count - 1 - k
            if n_valid <= 0:
                break 
            
            high = high_arr[k+1:k+n_valid+1]
            low = low_arr[k+1:k+n_valid+1]
            
            stop_loss = slice(0, n_valid)
            
            hit_take_profit = (
                (high >= take_profit_line[stop_loss]) 
                & (first_take_profit[stop_loss] == big)
            )
            hit_stop_loss = (
                (low <= stop_loss_line[stop_loss])
                & (first_stop_loss[stop_loss] == big)
            )
            
            first_take_profit[stop_loss][...] = \
                np.where(hit_take_profit, k, first_take_profit[stop_loss])
            first_stop_loss[stop_loss][...] = \
                np.where(hit_stop_loss, k, first_stop_loss[stop_loss])
                
        # BUY 조건: 익절이 손절보다 '엄격히' 먼저 (동시 도달 → 손절 우선 → HOLD)
        buy_mask = first_take_profit < first_stop_loss
        label_arr[:-1][buy_mask] = cfg.data.dir2idx[cfg.str.buy]
        label_T = TripleBarrierLabeler.label_distribution(label_arr)
        logger.point(msg.feature.labeling_result(bars, label_T))
        return label_arr
    
    @staticmethod
    def label_distribution(labels: np.ndarray) -> dict[SignalDirection, int]:
        """ 라벨 분포 확인 """
        return {
            direction: int((labels == idx).sum())
                for direction, idx in cfg.data.dir2idx.items()
        }
        