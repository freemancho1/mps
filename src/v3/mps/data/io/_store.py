""" 
LocalParquetStore ─ DataStorePort의 구현체(KIS API 미 연결 시 사용)

[역할]
  - 분봉 데이터를 로컬 Parquet 파일로 저장 또는 파일에서 불러옴
  - pykrx로 수집한 데이터를 캐싱하여 반복 실행 시 재수집을 방지함.
  
[교체 계획]
  - KIS API 연계 시 TimescaledDB(또는 InfluxDB)로 교체하면서 변경 예정
  - DataStorePort 인터페이스(mps.core.ports)를 유지하면,
    이 파일만 교체해도 상위 레이어(HistoricalDataLoader)는 수정 불요
    
[파일 구조]
  - artifacts/store/{ticker}/minute_bars.parquet
    → timestamp, index, OHLCV 컬럼값
"""
from __future__ import annotations 

import pandas as pd 
from pathlib import Path
from datetime import datetime 
from typing import cast, Any, Optional

from mps.config import cfg, msg 
from mps.core.types import Bar
from mps.core.libs import to_float, to_int
from mps.freelibs import logger


class LocalParquetStore:
    """ 종목별 분봉 데이터를 parquet 파일로 입출력 수행 """
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir = base_dir or cfg.path.store 
        
    def _gen_store_fpath(self, ticker: str) -> Path:
        # 종목별로 별도의 파일에 저장
        file_path = self._base_dir / f"{ticker}_{cfg.path.store_fname}"
        return file_path
    
    def load_bars(self, ticker: str, start: datetime, end: datetime) -> list[Bar]:
        """ 
        지정됨 날짜 구간의 Bar 리스트를 parquet에서 읽어 반환.
        
        - 파일이 없으면 빈 리스트를 반환 → HistoricalDataLoader에서 수집행위를 하는 트리거로 사용
        - timestamp 마스킹으로 불필요한 메모리 사용 줄임
        - 반환된 Bar들은 모두 is_complete=True ─ 저장 시점에 완성된 봉만 사용
        """
        store_fpath = self._gen_store_fpath(ticker)
        if not store_fpath.exists():
            logger.error(msg.pp.store.file_not_found(store_fpath))
            return []
        
        df = pd.read_parquet(store_fpath)
        df.index = pd.to_datetime(df.index)
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        sub_df = df.loc[mask]
        logger.debug(msg.pp.store.load_store_info(sub_df))

        result: list[Bar] = []
        for row in sub_df.itertuples():
            result.append(Bar(
                ticker=ticker,
                timestamp=cast(pd.Timestamp, row.Index).to_pydatetime(),
                open=to_float(row.open),
                high=to_float(row.high),
                low=to_float(row.low),
                close=to_float(row.close),
                volume=to_int(row.volume),
                is_complete=True
            ))

        return result
        
    def save_bars(self, bars: list[Bar]) -> None:
        """ 
        Bar 리스트를 parquet 파일로 저장

        [데이터 저장 원칙]
          - 종목당 각각 파일을 작성하며, 동일 timestamp 데이터는 새 데이터로 갱신
            → 재수집 시 최신 데이터로 업데이트
        """
        if not bars: 
            return 
        
        # Bar dataclass → Dataframe (필드명 그대로 컬럼이 됨)
        df = pd.DataFrame([vars(bar) for bar in bars])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        ticker = bars[0].ticker
        store_fpath = self._gen_store_fpath(ticker)
        if store_fpath.exists():
            old_df = pd.read_parquet(store_fpath)
            df = pd.concat([old_df, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(store_fpath)
        logger.debug(msg.pp.store.save_result(ticker, df))
