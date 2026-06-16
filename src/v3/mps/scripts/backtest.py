""" Phase-2 백테스트 실행 스크립트 """
from __future__ import annotations 

import argparse
from datetime import date, datetime 

from mps.config import cfg, msg 
from mps.core.types import PerformanceReport
from mps.data.io import LocalParquetStore, HistoricalDataLoader
from mps.trade.backtest import WalkForwardValidator
from mps.freelibs import logger 


def main():
    args = parse_args()
    logger.info(msg.bt.title(args))
    
    ticker, start, end, capital, train_days, test_days = \
        args.ticker, args.start, args.end, args.capital, args.train_days, args.test_days
        
    start_date = date(int(start[:4]), int(start[4:6]), int(start[6:]))
    end_date = date(int(end[:4]), int(end[4:6]), int(end[6:]))
    
    store = LocalParquetStore()
    loader = HistoricalDataLoader(store=store)
    
    _, bars = loader.load(ticker, start_date, end_date)
    if not bars:
        logger.error(msg.bt.err.no_data)
        return 
    
    # ── Walk-Forward 검증 ────────────────────
    
    start_dt = datetime.now()
    
    validator = WalkForwardValidator(train_days, test_days, capital)
    report: list[PerformanceReport] = validator.run(bars)
    # TODO 0616-1003: WalkForwardValidator 작업 후
    
    logger.info(msg.bt.result(datetime.now() - start_dt))


def parse_args():
    p = argparse.ArgumentParser(description=msg.bt.script_title)
    p.add_argument(cfg.key.ticker, default=cfg.run.tickers[0])
    p.add_argument(cfg.key.start, default=cfg.run.start_date_str)
    p.add_argument(cfg.key.end, default=cfg.run.end_date_str)
    p.add_argument(cfg.key.capital, type=float, default=cfg.run.init_capital)
    # [이후 답변]: 여기는 백테스트 구간인데, 왜 아래 2개의 인자값을 받아야 하나?
    #              아래 인자는 백테스트 영역이 아니고, 모델 훈련 영역인거 같은데?
    p.add_argument(cfg.key.train_days, type=int, default=cfg.run.train_days)
    p.add_argument(cfg.key.test_days, type=int, default=cfg.run.test_days)
    
    return p.parse_args()


if __name__ == "__main__":
    main()