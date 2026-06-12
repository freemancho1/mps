""" 
HistoricalDataLoader ─ 과거 분봉 데이터 수집 및 로컬 캐싱

[수집 전략]
  1순위: LocalParquetStore 캐시 (force_refresh=True면 실행하지 않음)
  2순위: KIS REST API (KIS_APP_KEY 설정 시 ─ 현재 설정하지 않음)
  3순위: pykrx 일봉 기반 합성 분봉 (개발·테스트 전용)

[합성 데이터의 한계]
  - GBM 기반 합성 분봉에는 "학습 가능한 패턴"이 없음.
  - 모델 성능·수익성 수치는 파이프라인 검증용일 뿐, 
    전략 검증은 실제 KIS 분봉 데이터 확보 후에만 의미가 있음.
"""
from __future__ import annotations 

import numpy as np 
import pandas as pd 
from datetime import date, datetime, timedelta 
from typing import Optional

from mps.config import cfg, msg
from mps.core.types import Bar 
from mps.core.calendar import market_open_datetime
from mps.data.io import LocalParquetStore
from mps.core.libs import to_float, to_int


class HistoricalDataLoader:
    def __init__(self, store: Optional[LocalParquetStore] = None) -> None:
        self._store: LocalParquetStore = store or LocalParquetStore()

    def load(
        self,
        ticker: str,
        start_date: date, end_date: date,
        force_refresh: Optional[bool] = None,
    ) -> tuple[str, list[Bar]]:
        """ 
        캐시 우선 로드 ─ 캐시가 없거나 force_refresh=True면 새로 수집

        [흐름]
          1. start_date ~ end_date 범위 캐시 조회
          2. 캐시가 있고 force_refresh가 아니면 즉시 반환
          3. 아니면 수집(또는 합성) 후 저장·반환
        """
        if force_refresh is None:
            force_refresh = cfg.data.force_refresh_data

        # 시작일 00:00:00 ~ 종료일 23:59:59
        start_datetime = datetime.combine(start_date, datetime.min.time(), tzinfo=cfg.sys.timezone)
        end_datetime = datetime.combine(end_date, datetime.max.time(), tzinfo=cfg.sys.timezone)

        cached_bars = self._store.load_bars(ti)
