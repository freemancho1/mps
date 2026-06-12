""" 
Phase-2 인공지능 기반 매도 진입점 예측 모델 학습

[작업 흐름]
  1. 과거 봉 데이터 로드 (backtest.py와 동일한 로드 ─ 데이터가 없으면 합성)
  2. TripleBarrier 라벨 생성 → 수치·패턴 트랙 Dataset 구성
  3. LSTM(수치)·1D-CNN(패턴) 각각 학습 (시간순 train/val, 조기 종료)
  4. 가중치를 레지스트리 경로에 저장 (메타 데이터 포함)
  
[재현 가능성]
  - cfg.sys.seed 고정
  - 동일 데이터·코드·시드 → 동일 가중치 → 동일한 재현
"""
from __future__ import annotations

import argparse 
import torch 
from datetime import datetime 
from pathlib import Path

from mps.config import cfg, msg
from mps.freelibs import logger


def main() -> None:
    p = argparse.ArgumentParser(description=msg.training.title)
    p.add_argument(cfg.key.ticker, default=cfg.run.tickers[0])
    p.add_argument(cfg.key.start, default=cfg.run.start_date_str)
    p.add_argument(cfg.key.end, default=cfg.run.end_date_str)
    args = p.parse_args()
    logger.debug(msg.training.info(msg.training.title, args.ticker, args.start, args.end))
    
    # TODO 0: LocalParquetStore 작업 후

def test() -> None:
    print("My test...")
    

if __name__ == "__main__":
    main()