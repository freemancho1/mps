""" 
Phase-2 인공지능 기반 매매 진입점 ─ 모델 학습 

[작업 흐름]
  1. 과거 봉 데이터 로드 (backtest.py와 동일한 로드 ─ 데이터가 없으면 합성)
  2. Triple Barrier 라벨 생성 → 수치·패턴 트랙 Dataset 구성
  3. LSTM(수치)·1D-CNN(패턴) 각각 학습 (시간순 train/val, 조기 종료)
  4. 가중치를 레지스트리 경로에 저장 (메타데이터 포함)

[재현 가능성]
  - cfg.run.seed 고정. 
  - 동일 데이터·코드·시드 → 동일 가중치 → 동일한 재현
"""
from __future__ import annotations 

import numpy as np 
import argparse 
from datetime import date, datetime, timedelta 
from zoneinfo import ZoneInfo 

from mps.config import cfg, msg
from mps.core.types import Bar


# TODO 1: 여기 수행 필요
def load_bars(ticker: str, start: str, end: str) -> list[Bar]:
    print(f"Run load_bars... ticker[{ticker}], date={start}~{end}")
    return []


def main() -> None:
    p = argparse.ArgumentParser(description=msg.tm.title)
    p.add_argument(cfg.key.ticker, default=cfg.run.tickers[0])
    p.add_argument(cfg.key.start, default=cfg.run.start_date_str)
    p.add_argument(cfg.key.end, default=cfg.run.end_date_str)
    args = p.parse_args()

    bars = load_bars(args.ticker, args.start, args.end)


if __name__ == "__main__":
    main()
