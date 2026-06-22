""" 
LocalParquetStore ─ DataStorePort의 구현체(KIS API 미 연결 시 사용)

[역할]
  - 분봉 데이터를 로컬 Parquet 파일로 저장 또는 파일에서 불러옴.
  - pykrx로 수집한 데이터를 캐싱해서 반복 실행 시 재수집을 방지함.

[교체 계획]
  - KIS API 연계시 TimescaledDB(또는 InfluxDB)로 교체하면서 변경 예정
  - DataStorePort 인터페이스를 유지하면 상위 레이어 수정 불요
"""
from __future__ import annotations 

import pandas as pd 
from pathlib import Path 
from datetime import datetime 
from typing import cast, Optional 

from mps.core.config import cfg, msg 
from mps.core.types import Bar 
from mps.core.libs import to_float, to_int
from mps.core.libs import logger 


class LocalParquetStore:
    """ 종목별 분봉 데이터를 parquet 파일로 입출력 수행 """
    def __init__(
        self, 
        base_dir: Optional[Path] = None,
        file_name: Optional[str] = None,
    ) -> None:
        self._base_dir: Path = cfg.path.store if base_dir is None else base_dir 
        self._file_name: str = cfg.path.store_fname if file_name is None else file_name 

    def _build_store_path(self, ticker: str) -> Path:
        # 종목별로 별도의 파일에 저장
        return self._base_dir / f"{ticker}_{self._file_name}"
    
    def load_bars(self, ticker: str, start_dt: datetime, end_dt: datetime) -> list[Bar]:
        """ 
        지정된 날짜 구간의 Bar 리스트를 Parquet에서 읽어 반환.

        - 파일이 없으면 빈 리스트를 반환 
          → HistoricalDataLoader에서 수집 행위를 하는 트리거로 사용
        - timestamp 마스킹으로 불필요한 메모리 사용 줄임.
        - 반환된 Bar들은 모두 is_complete=True 
          → 저장 시점에서 완성된 봉만 사용 (look-ahead 준수)
        """
        store_path = self._build_store_path(ticker)
        if not store_path.exists():
            logger.warning(msg.store.file_not_found_err(store_path))
            return []
        
        source_df = pd.read_parquet(store_path)
        source_df.index = pd.to_datetime(source_df.index)

        mask = (source_df.index >= pd.Timestamp(start_dt)) \
               & (source_df.index <= pd.Timestamp(end_dt))
        sub_df = source_df.loc[mask]
        logger.debug(msg.store.load_data_info(sub_df))

        results: list[Bar] = []
        for row in sub_df.itertuples():
            result: Bar = Bar(
                ticker=ticker,
                timestamp=cast(pd.Timestamp, row.Index).to_pydatetime(),
                open=to_float(row.open),
                high=to_float(row.high),
                low=to_float(row.low),
                close=to_float(row.close),
                volume=to_int(row.volume),
                is_complete=True,
            )
            results.append(result)
        return results
    
    def save_bars(self, bars: list[Bar]) -> None:
        """ 
        Bar 리스트를 Parquet 파일로 저장

        종목당 각각 파일을 작성하며, 동일 timestamp 데이터는 새 데이터로 갱신
        """
        if not bars: 
            return
        
        # list[Bar] → DataFrame: 필드명 그대로 컬럼이 됨
        new_df = pd.DataFrame([vars(bar) for bar in bars])
        new_df[cfg.key.timestamp] = pd.to_datetime(new_df[cfg.key.timestamp])
        new_df = new_df.set_index(cfg.key.timestamp).sort_index()

        ticker = bars[0].ticker 
        store_path = self._build_store_path(ticker)
        if store_path.exists():
            old_df = pd.read_parquet(store_path)
            new_df = pd.concat([old_df, new_df])
            # 중복된 일자(인덱스) 정보는 마지막 데이터로 업데이트함.
            new_df = new_df[~new_df.index.duplicated(keep=cfg.key.last)].sort_index()

        new_df.to_parquet(store_path)
        logger.debug(msg.store.save_parquet_info(new_df))
