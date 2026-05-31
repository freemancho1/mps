""" 
LocalParquetStore ─ DataStorePort의 Phase-1 구현체

[역할]
  - 분봉 데이터를 로컬 Parquet 파일로 저장 또는 파일에서 불러옴
  - pykrx로 수집한 데이터를 캐싱하여 반복 실행 시 재수집을 방지함.

[교체 계획]
  - Phase-2에서 TimescaleDB(또는 InfluxDB)로 교체 예정
  - DataStorePort 인터페이스(core/ports.py)를 유지하면,
    이 파일만 교체해도 상위 레이어(HistoricalDataLoader)는 수정 불필요

[파일 구조]
  - mps/data/store/{ticker}/minute_bars.parquet → timestamp index, OHLCV 컬럼값
"""
from __future__ import annotations

import pandas as pd 
from datetime import datetime 
from pathlib import Path 
from typing import Any, cast 

from mps.core.types import Bar 
from mps.config import cfg, msg 


class LocalParquetStore:
    """ 종목별 분봉 데이터를 parquet 파일로 저장·읽기 작업 수행 """
    def __init__(self, base_dir: Path = cfg.store.dir) -> None: 
        self._base_dir: Path = base_dir

    def _ticker_path(self, ticker: str) -> Path:
        """ 종목별 디렉토리 생성 후 parquet 파일 경로 반환 """
        fpath = self._base_dir / ticker 
        fpath.mkdir(parents=True, exist_ok=True)
        fpath = fpath / cfg.store.fname 
        return fpath
    
    def load_bars(self, ticker: str, start_dt: datetime, end_dt: datetime) -> list[Bar]:
        """ 
        지정된 날짜 구간의 Bar 리스트를 parquet에서 읽어 반환.
        
        - 파일이 없으면 빈 리스트를 반환 → HistoricalDataLoader의 수집행위 트리거로 활용
        - timestamp 마스킹으로 불필요한 메모리 사용 줄임
        - 반환된 Bar들은 모두 is_complete=True(저장 시점에 완성된 봉만 사용)
        """
        fpath = self._ticker_path(ticker)
        if not fpath.exists():
            print(msg.data.file_not_found(fpath))
            return []
        
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        # 구간 필터링
        mask = (df.index >= pd.Timestamp(start_dt)) & \
               (df.index <= pd.Timestamp(end_dt))
        sub_df = df.loc[mask]
        print(msg.data.load_info(start_dt, end_dt, sub_df))

        result: list[Bar] = []
        for row in sub_df.itertuples():
            result.append(Bar(
                ticker=ticker,
                timestamp=cast(pd.Timestamp, row.Index).to_pydatetime(),
                open=float(cast(Any, row.open)),
                high=float(cast(Any, row.high)),
                low=float(cast(Any, row.low)),
                close=float(cast(Any, row.close)),
                volume=int(cast(Any, row.volume)),
                is_complete=True,
            ))
        print(msg.data.result_info(result))
        return result 
    
    def save_bars(self, bars: list[Bar]) -> None:
        """ 
        Bar 리스트를 parquet 형식으로 저장 (기존 데이터와 병합되면, 중복 제거)

        중복 처리 전략: 동일 timestamp가 있으면 새 데이터(keep="last")를 우선함.
                       → pykrs 재수집 시 최신 데이터로 덮어쓰는 것을 허용함.
        """
        if not bars:
            return 
        
        # Bar dataclass → DataFrame으로 변환 (필드명 유지)
        df = pd.DataFrame([vars(bar) for bar in bars])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # DataFrame 업데이트
        ticker = bars[0].ticker 
        fpath = self._ticker_path(ticker)
        if fpath.exists():
            old_df = pd.read_parquet(fpath)
            combined_df = pd.concat([old_df, df])
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]\
                .sort_index()
            combined_df.to_parquet(fpath)
        else:
            df.to_parquet(fpath)
