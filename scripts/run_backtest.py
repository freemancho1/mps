from __future__ import annotations

from datetime import date 
from pathlib import Path 
import argparse
import sys 

from mps.data.store import LocalParquetStore
from mps.data.loader import HistoricalDataLoader
from mps.sys.config import settings


def main():

    args = parse_args()
    # 문자열을 날짜 형식으로 변환
    start_date = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:]))
    end_date = date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:]))

    # --- 실행 정보 요약 ------------------------------------------
    print(
        f"\n{'='*60}\n"
        f"○ MPS Phase 1 Back-Test \n"
        f"   - 종목코드: {args.ticker} \n"
        f"   - 기간: {start_date} ~ {end_date} \n"
        f"   - 초기자본: {args.capital:,.0f} 원 \n"
        f"   - 보수적 왕복 비용: {settings.cost.roundtrip_cost:.2%} \n"
        f"{'='*60}\n"
    )

    # --- 단계1: 데이터 로드 --------------------------------------
    # LocalParquetStore 캐시 → 없으면 pykrx 일봉으로 합성 분봉 생성
    # 반환값: list[Bar] - timestamp 순 정렬, is_complete=True
    print("데이터 로드 중 ...")
    store = LocalParquetStore()
    loader = HistoricalDataLoader(store)
    bars = loader.load(args.ticker, start_date, end_date)
    print(f"  - 총 {len(bars):,}봉 로드 완료")

    if not bars:
        print("데이터가 존재하지 않습니다. pykrx 설치하거나 네트워크 연결을 확인하세요.")
        return
    
    # --- 단계2: Walk-Forward 검증 --------------------------------
    # 학습 60거래일 + 테스트 10거래일 슬라이딩 윈도우 반복
    # 각 윈도우마다 독립 HistoricalSimulator를 생성하여 PerformanceReport 반환
    # 셔플없이 시간순으로 처리하며, 여러 구간의 평균 성과로 과적합 여부 판단.
    print("Walk-Forward 검증 실행 중 ...")
    # TODO 1: WalkForwardValidator() 처리후 계속


def parse_args():
    p = argparse.ArgumentParser(description="MPS Phase1 백테스트")
    p.add_argument("--ticker", default="005930", help="종목 코드 (기본값: 삼성전자)")
    p.add_argument("--start", default="20250101", help="시작일 YYYYMMDD (기본값: 20250101)")
    p.add_argument("--end", default="20251231", help="종료일 YYYYMMDD (기본값: 20251231)")
    p.add_argument("--capital", type=float, default=10_000_000.0, help="초기 자본 (기본값: 10,000,000.0원)")
    return p.parse_args()

if __name__ == "__main__":
    main()