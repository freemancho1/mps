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
from mps.core.types import Bar
from mps.data.io import LocalParquetStore, HistoricalDataLoader
from mps.data.features import TripleBarrierLabeler, TripleBarrierDataset
from mps.models.numeric.lstm import LSTMNet
from mps.freelibs import logger


def load_bars(ticker: str, start: str, end: str) -> list[Bar]:
    start_date = datetime.strptime(start, cfg.sys.date_format).date()
    end_date = datetime.strptime(end, cfg.sys.date_format).date()
    # logger.debug(msg.training.load_data_info(ticker, start_date, end_date))

    store = LocalParquetStore()
    loader = HistoricalDataLoader(store=store)

    _, bars = loader.load(ticker, start_date, end_date)
    return bars

def train_track(bars: list[Bar], track: str, model: torch.nn.Module, save_path: Path) -> None:
    logger.debug(msg.training.model_info(track, model))
    labeler = TripleBarrierLabeler()
    dataset = TripleBarrierDataset(bars, track, labeler=labeler)
    dist = dict(zip([cfg.str.buy, cfg.str.hold], dataset.class_counts().tolist()))
    logger.debug(msg.pp.features.dataset_result(dataset, dist))

    # TODO 0614-1736: ModelTrainer() 작업 후


def main() -> None:
    p = argparse.ArgumentParser(description=msg.training.title)
    p.add_argument(cfg.key.ticker, default=cfg.run.tickers[0])
    p.add_argument(cfg.key.start, default=cfg.run.start_date_str)
    p.add_argument(cfg.key.end, default=cfg.run.end_date_str)
    args = p.parse_args()
    logger.debug(msg.training.info(msg.training.title, args.ticker, args.start, args.end))
    
    bars = load_bars(args.ticker, args.start, args.end)

    logger.debug(msg.training.start)
    start_datetime = datetime.now()

    train_track(
        bars=bars,
        track=cfg.model.numeric_track,
        model=LSTMNet(**cfg.train.lstm_settings.to_dict()),
        save_path=cfg.path.lstm_model_fpath
    )

    logger.debug(msg.training.finished(datetime.now() - start_datetime))


if __name__ == "__main__":
    main()