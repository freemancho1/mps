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
from datetime import date

from mps.sys import config as cfg


def main():
    args = parse_args()
    
    start_date = date(int(args.start[:4]), int(args.start[5:7]), int(args.start[8:]))
    end_date = date(int(args.end[:4]), int(args.end[5:7]), int(args.end[8:]))

    # ── 테스트 정보 요약 ─────────────────────────
    _args = {
        "title": cfg.run.msg.title, 
        "ticker": args.ticker,
        "start": start_date,
        "end": end_date,
        "capital": args.capital,
        "roundtrip_cost": cfg.cost.roundtrip_cost
    }
    print(cfg.run.msg.summary(_args))

    # ── 데이터 로드 ────────────────────────────
    


def parse_args():
    p = argparse.ArgumentParser(description=cfg.run.msg.title)
    p.add_argument(
        cfg.run.key.ticker,             # 종목코드
        default=cfg.run.tickers[0],     # 기본값: 삼성전자 005930
        help=cfg.run.msg.ticker
    )
    p.add_argument(
        cfg.run.key.start_date,         # 테스트 시작일
        default=cfg.run.start_date,     # 기본값: 2025-01-01
        help=cfg.run.msg.start_date
    )
    p.add_argument(
        cfg.run.key.end_date,           # 테스트 종료일
        default=cfg.run.end_date,       # 기본값: 2025-12-31
        help=cfg.run.msg.end_date
    )
    p.add_argument(
        cfg.run.key.capital,            # 초기 투자 금액
        type=float, 
        default=cfg.run.capital,        # 기본값: 10,000,000 원
        help=cfg.run.msg.capital
    )
    return p.parse_args()


if __name__ == "__main__":
    main()