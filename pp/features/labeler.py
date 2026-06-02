""" 
TripleBarrierLabeler ─ Triple Barrier 라벨 생성기 (결정6)

[역할]
  - 학습 기반 모델(LSTM·1D-CNN)의 정답(target) 라벨을 생성.
  - 각 봉 t에서 진입했다고 가정하고, 이후 time_horizon분 동안:
    · 고가가 익절선(+take_profit) 먼저 도달 → 상방 우위     → BUY
    · 저가가 손절선(-stop_loss) 먼저 도달   → 하방 우위     → SELL
    · 둘 다 미도달 (시간 만료)              → 방향 불명확   → HOLD

[결정6 근거]
  - 실 거래는 "방향이 맞는가"가 아니라,
    "익절·손절·시간만료 중 무엇이 먼저 도달했는가의 문제로,
    수치·패턴 두 트랙이 동일 라벨을 공유하므로 두 출력을 직접 비교·결합할 수 있음.

[보수성]
  - 한 봉의 (저·고가)가 손·익절선을 동시에 포함하면,
    어느 쪽이 먼저 닿았는지 분봉만으로 알 수 없음.
    StopLossTakeProfitGuard와 동일하게 '손절(하방) 우선'으로
    라벨링해 백테스트 청산 로직과 일관성을 유지.

[클래스 인덱스]
  - 학습 시 CrossEntropyLoss를 쓰므로 정수 인덱스로 매핑한다.
    BUY=0, SELL=1, HOLD=2
"""
from __future__ import annotations 

import numpy as np 

from mps.config import cfg, msg
from mps.core.types import Bar, Direction


# 학습용 클래스 인덱스 ⇔ 방향 매칭 (모델 출력 해석에 공용)
LABEL_TO_IDX: dict[Direction, int] = {"BUY": 0, "SELL": 1, "HOLD": 2}
IDX_TO_LABEL: dict[int, Direction] = {0: "BUY", 1: "SELL", 2: "HOLD"}


class TripleBarrierLabeler:
    def __init__(
        self,
        take_profit: float = cfg.run.take_profit,   # 0.02  (= 2.0%)
        stop_loss: float = cfg.run.stop_loss,       # 0.005 (= -0.5%)
        time_horizon: int = cfg.run.time_horizon,   # 60    (= 60분)
    ) -> None:
        self._take_profit = take_profit 
        self._stop_loss = stop_loss 
        self.time_horizon = time_horizon

    def label(self, bars: list[Bar]) -> np.ndarray:
        """ 
        각 봉의 Triple Barrier 라벨(클래스 인덱스)을 반환.

        반환: shape [len(bars)] int64 배열.
              ─ 마지막 horizon개 봉은 미래 데이터가 부족해 HOLD로 채워지므로,
                 학습 시에는 Dataset에서 제외
        """
        num = len(bars)
        print(msg.features.label_size(num))

        closes = np.fromiter((bar.close for bar in bars), dtype=np.float64, count=num)
        highs = np.fromiter((bar.high for bar in bars), dtype=np.float64, count=num)
        lows = np.fromiter((bar.low for bar in bars), dtype=np.float64, count=num)

        labels = np.full(num, LABEL_TO_IDX["HOLD"], dtype=np.int64)

        for pit in range(num):  # pit: Point in Time
            entry = closes[pit]
            take_profit_line = entry * (1.0 + self._take_profit)
            stop_loss_line = entry * (1.0 - self._stop_loss)
            end = min(pit + self.time_horizon, num - 1)

            outcome: Direction = "HOLD"
            for idx in range(pit + 1, end + 1):
                hit_stop_loss = lows[idx] <= stop_loss_line 
                hit_take_profit = highs[idx] >= take_profit_line
                if hit_stop_loss:
                    outcome = "SELL"
                    break 
                if hit_take_profit:
                    outcome = "BUY"
                    break 
            labels[pit] = LABEL_TO_IDX[outcome]

        return labels 
    
    def label_distribution(self, labels: np.ndarray) -> dict[Direction, int]:
        """ 라벨 분포 확인용 (클래스 불균형 진단) """
        return {
            direction: int((labels == idx).sum())
                for direction, idx in LABEL_TO_IDX.items()
        }