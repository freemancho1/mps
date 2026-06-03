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
import torch
from datetime import datetime
from pathlib import Path

from models.registry import save_checkpoint
from mps.pp.dataio.store import LocalParquetStore
from mps.pp.dataio.loader import HistoricalDataLoader
from mps.pp.features.labeler import TripleBarrierLabeler
from mps.pp.features.dataset import TripleBarrierDataset
from mps.models.numeric.lstm import LSTMNet
from mps.models.pattern.cnn import CNN1DNet
from mps.models.trainer import ModelTrainer
from mps.config import cfg, msg
from mps.core.types import Bar


def load_bars(ticker: str, start: str, end: str) -> list[Bar]:
    store = LocalParquetStore()
    loader = HistoricalDataLoader(store)
    start_date = datetime.strptime(start, cfg.run.date_format).date()
    end_date = datetime.strptime(end, cfg.run.date_format).date()
    print(msg.tm.info(ticker, start_date, end_date))
    _, bars = loader.load(ticker, start_date, end_date)
    return bars


def train_track(
    bars: list[Bar], 
    track: str, 
    model: torch.nn.Module, 
    save_path: Path
) -> None:
    print(msg.training.track_title(bars, track, model, save_path))
    labeler = TripleBarrierLabeler()
    dataset = TripleBarrierDataset(bars, track, labeler=labeler)
    dist = dict(zip([cfg.key.BUY, cfg.key.SELL, cfg.key.HOLD], dataset.class_counts().tolist()))
    print(msg.training.sample_labels(dataset, dist))
    
    trainer = ModelTrainer()
    model, history = trainer.train(model, dataset)
    print(msg.training.result(history))

    meta = {
        "track": track,
        "seed": cfg.train.seed,
        "arch": model.__class__.__name__,
        "label_dist": dist, 
        "best_val_loss": round(history.best_val_loss, 5),
        "best_val_acc": history.val_acc[history.best_epoch],
        "take_profit": cfg.run.take_profit,
        "stop_loss": cfg.run.stop_loss,
        "horizon": cfg.run.time_horizon,
    }
    path = save_checkpoint(model, save_path, meta)
    print(msg.training.save_model_info(meta))


def main() -> None:
    p = argparse.ArgumentParser(description=msg.tm.title)
    p.add_argument(cfg.key.ticker, default=cfg.run.tickers[0])
    p.add_argument(cfg.key.start, default=cfg.run.start_date_str)
    p.add_argument(cfg.key.end, default=cfg.run.end_date_str)
    args = p.parse_args()

    bars = load_bars(args.ticker, args.start, args.end)

    start_time = datetime.now()

    train_track(
        bars=bars, 
        track=cfg.run.numeric_track,
        model=LSTMNet(**cfg.lstm.to_dict()),
        save_path=cfg.model.lstm_model_fpath
    )
    train_track(
        bars=bars,
        track=cfg.run.pattern_track,
        model=CNN1DNet(**cfg.cnn.to_dict()),
        save_path=cfg.model.cnn_model_fpath
    )

    print(msg.training.finished(datetime.now() - start_time))


if __name__ == "__main__":
    main()
