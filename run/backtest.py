""" 
Phase-2 백테스트 실행 스크립트

[사용법]
  python run_backtest.py [--ticker 005930] [--start 20250101] [--end 20251231]
"""
from __future__ import annotations

import argparse 
from datetime import date, datetime 

from mps.config import cfg, msg 
from mps.pp.dataio.store import LocalParquetStore
from mps.pp.dataio.loader import HistoricalDataLoader


def main():
    args = parse_args()
    print(msg.bt.args_info(args))

    ticker, start, end, capital = args.ticker, args.start, args.end, args.capital
    start_date = date(int(start[:4]), int(start[4:6]), int(start[6:]))
    end_date = date(int(end[:4]), int(end[4:6]), int(end[6:]))

    store = LocalParquetStore()
    loader = HistoricalDataLoader()
    load_from, bars = loader.load(ticker, start_date, end_date)
    if not bars:
        print(msg.bt.dataload_err)
        return 
    
    print(msg.bt.dataload_info(load_from, bars))

    start_dt = datetime.now()

    # ── Walk-Forward 검증 ───────────────────
    # TODO 0: WalkForwardValidator() 작업하면서 진행





def parse_args():
    p = argparse.ArgumentParser(description=msg.bt.title)
    p.add_argument(cfg.key.ticker, default=cfg.run.tickers[0])
    p.add_argument(cfg.key.start, default=cfg.run.start_date_str)
    p.add_argument(cfg.key.end, default=cfg.run.end_date_str)
    p.add_argument(cfg.key.capital, type=float, default=cfg.run.init_capital)

    return p.parse_args()


if __name__ == "__main__":
    main()