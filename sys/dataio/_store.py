""" 
LocalParquetStore ─ DataStorePort의 Phase-1 구현체

[역할]
  - 분봉 데이터를 로컬 Parquet 파일로 저장·로드함.
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
from typing import Optional, Any, cast 

from mps.sys.core.types import Bar
from mps.sys import cfg, msg


class LocalParquetStore:
    """ 종목별 분봉 데이터를 Parquet 파일로 저장·로드 """
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir = base_dir or cfg.store.dir
        
        print(msg.store.init_info(base_dir, self._base_dir))
        
    def _fpath(self, ticker: str) -> Path:
        """ 종목별 디렉토리를 생성하고 Parquet 파일 경로로 반환 """
        p = self._base_dir / ticker
        p.mkdir(parents=True, exist_ok=True)
        
        fpath = p / cfg.store.fname
        print(msg.store.fpath(fpath))
        return fpath
    
    def load_bars(self, ticker: str, start: datetime, end: datetime) -> list[Bar]:
        """ 
        지정 구간의 Bar 리스트를 Parquet에서 읽어 반환.
        - 파일이 없으면 빈 리스트 반환 → HistoricalDataLoader의 수집행위 트리거
        - timestamp 마스킹으로 불필요한 메모리 사용을 줄임.
        - 반환된 Bar 들은 모두 is_complete=True (저장 시점에 완성된 봉만 저장)
        """
        fpath = self._fpath(ticker)
        if not fpath.exists():
            print(msg.store.fpath_not_found(fpath))
            return []
        
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        # 구간 필터링 : start ~ end
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        sub = df.loc[mask]
        
        print(msg.store.load_bars.dates(start, end, mask))
        print(msg.store.load_bars.size(df))
        
        result: list[Bar] = []
        for row in sub.itertuples():
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
            
        print(msg.store.load_bars.return_size(result))
        return result

    def save_bars(self, bars: list[Bar]) -> None:
        """ 
        Bar 리스트를 Parquet 형식으로 저장 (기존 데이터와 병합되면, 중복 제거)
        
        중복 처리 전략: 동일 timestamp가 있으면 새 데이터(keep="last")를 우선함.
                       ⇒ pykrx 재수집 시 최신 데이터로 덮어쓰는 것을 허용함.
        """
        if not bars:
            return 
        
        ticker = bars[0].ticker 
        
        # Bar dataclass → DataFrame으로 변환 (필드명 유지)
        new_df = pd.DataFrame([vars(bar) for bar in bars])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
        new_df = new_df.set_index("timestamp").sort_index()
        
        fpath = self._fpath(ticker)
        if fpath.exists():
            # 기존 데이터가 있으면 시간순으로 다시 저장(같은 시간대는 나중 데이터로)
            old_df = pd.read_parquet(fpath)
            combined = pd.concat([old_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined.to_parquet(fpath)
        else:
            new_df.to_parquet(fpath)
    