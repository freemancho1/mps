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

from mps.settings import cfg, msg 
from mps.core.types import Bar


class LocalParquetStore:
    """ 종목별 분봉 데이터를 parquet 파일로 입출력 수행 """
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir = base_dir or cfg.path.store 
        
    def _gen_store_path(self, ticker: str) -> Path:
        # 종목별로 별도의 디렉토리에 저장
        ticker_dir = self._base_dir / ticker 
        ticker_dir.mkdir(parents=True, exist_ok=True)
        file_path = ticker_dir / cfg.path.store_fname
        return file_path
    
    def load_bars(self, ticker: str, start: datetime, end: datetime) -> list[Bar]:
        """ 
        지정됨 날짜 구간의 Bar 리스트를 parquet에서 읽어 반환.
        
        - 파일이 없으면 빈 리스트를 반환 → HistoricalDataLoader에서 수집행위를 하는 트리거로 사용
        - timestamp 마스킹으로 불필요한 메모리 사용 줄임
        - 반환된 Bar들은 모두 is_complete=True ─ 저장 시점에 완성된 봉만 사용
        """
        store_path = self._gen_store_path(ticker)
        if not store_path.exists():
            print(msg.pp.store.file_not_found(store_path))
            return []
        
        # TODO 1: logger 작업 후
        return []
        
