""" 
Phase 1 백테스트 실행 스크립트

[실행 흐름]
  1. 커맨드라인 인자 파싱 (argparse)
  2. 데이터 로드
  3. Walk-Forward 검증: 60일 학습 + 10일 테스트 슬라이딩 윈도우

[사용법]
  python run_backtest.py [--ticker 005930] [--start 20250101] [--end 20251231]
"""
from __future__ import annotations 

import sys 
import argparse 
from datetime import date, datetime

from mps.sys.dataio import LocalParquetStore, HistoricalDataLoader
from mps.sys import cfg, msg


def main():
    args = parse_args()

    # ── 테스트 정보 요약 ─────────────────────────
    _args = {
        "title": msg.run.info.title, 
        "ticker": args.ticker,
        "start": date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:])),
        "end": date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:])),
        "capital": args.capital,
        "roundtrip_cost": cfg.cost.roundtrip_cost
    }
    print(msg.run.info.summary(_args))
    print(msg.run.sys.summary(cfg.sys))

    # ── 1단계: 데이터 로드 ───────────────────────
    print(msg.run.data_load.title(datetime.now()))
    store = LocalParquetStore()
    loader = HistoricalDataLoader(store)
    load, bars = loader.load(args.ticker, _args["start"], _args["end"])
    print(msg.run.data_load.result(load, bars))
    
    if not bars:
        print(msg.run.data_load.result_error(load))
        return 
    
    # ── 2단계: Walk-Forward 검증 ─────────────────────
    # 학습 60거래일 + 테스트 10거래일 슬라이딩 윈도우를 반복
    # 각 윈도우마다 독립 HistoricalSimulator를 생성하여, PerformanceReport 반환
    # → 여러 구간의 평균 성과로 과적합 여부 판단
    print(msg.run.wf.title(datetime.now()))
    


def parse_args():
    p = argparse.ArgumentParser(description=msg.run.info.title)
    p.add_argument(
        cfg.run.key.ticker,             # 종목코드
        default=cfg.run.tickers[0],     # 기본값: 삼성전자 005930
        help=msg.run.info.ticker
    )
    p.add_argument(
        cfg.run.key.start_date,         # 테스트 시작일
        default=cfg.run.start_date,     # 기본값: 2025-01-01
        help=msg.run.info.start
    )
    p.add_argument(
        cfg.run.key.end_date,           # 테스트 종료일
        default=cfg.run.end_date,       # 기본값: 2025-12-31
        help=msg.run.info.end
    )
    p.add_argument(
        cfg.run.key.capital,            # 초기 투자 금액
        type=float, 
        default=cfg.run.capital,        # 기본값: 10,000,000 원
        help=msg.run.info.capital
    )
    return p.parse_args()


if __name__ == "__main__":
    main()