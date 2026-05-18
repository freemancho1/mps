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