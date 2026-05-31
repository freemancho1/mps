""" 
HistoricalDataLoader ─ 과거 분봉 데이터 수집 및 로컬 캐싱

[수집 전략]
  - 1순위: LocalParquetStore 캐시 → 동일 구간이 저장되어 있다면 재수집 생략
           ─ 단, cfg.run.force_data_refresh=True이면 무조건 재 수집
  - 2순위: KIS_APP_KEY 환경변수 설정 시 KIS REST API로 실제 분봉 조회(현재 미구현)
  - 3순위: pykrx 일봉 기반 합성 분봉 생성 (개발·테스트 용으로만 사용)
"""
from __future__ import annotations

import random 
import numpy as np 
import pandas as pd 
from pykrx import stock as krx 
from datetime import date, datetime, timedelta 
from typing import Optional, Any, cast 

from mps.config import cfg, msg 
from mps.core.types import Bar 
from mps.core.calendar import trading_days, market_open_dt
from mps.pp.dataio.store import LocalParquetStore


class HistoricalDataLoader:
    def __init__(self, store: Optional[LocalParquetStore] = None) -> None:
        self._store: LocalParquetStore = store or LocalParquetStore()

    def load(
        self,
        ticker: str, 
        start_date: date, 
        end_date: date,
        force_refresh: bool = cfg.run.force_data_refresh
    ) -> tuple[str, list[Bar]]:
        """ 
        캐시 우선 불러옴 ─ 캐시가 없거나 force_data_refresh=True이면 새로 생성

        [흐름]
          1. start_date ~ end_date 범위로 캐시 데이터 조회
          2. 캐시 데이터가 있으면 즉시 값 반환 (별로 수집이나 생성 없음)
          3. 캐시 데이터가 없거나 force_data_refresh=True이면 생성 후 반환
        """
        start_dt = datetime.combine(
            start_date, datetime.min.time(), tzinfo=cfg.run.timezone
        )
        end_dt = datetime.combine(
            end_date, datetime.max.time(), tzinfo=cfg.run.timezone
        )

        cached_bars = self._store.load_bars(ticker, start_dt, end_dt)
        if cached_bars and not force_refresh:
            return cfg.store.load_store, cached_bars    # "STORE", 데이터
        
        load_from, bars = self._fetch(ticker, start_date, end_date)
        # TODO 3: 이곳에 있는 def _fetch() 생성 후 작업

    def _fetch(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date,
    ) -> tuple[str, list[Bar]]:
        """ 데이터 획득(조회 또는 합성) """
        if cfg.kis.app_key:
            print(msg.data.fetch.from_kis)
            return self._fetch_kis(ticker, start_date, end_date)
        print(msg.data.fetch.from_pykrx)
        return self._fetch_synthetic(ticker, start_date, end_date)
    
    def _fetch_kis(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> tuple[str, list[Bar]]:
        """ 
        KIS REST API 분봉 조회 (stub - API키 설정 후 kis_client.py에서 구현)

        [구현 시 고려 사항]
          - KIS API는 한 번에 최대 100~200봉만 반환하기 때문에 날짜 범위를 쪼개서 반복 호출
          - OAuth 토큰 만료(약 24시간) 처리 필요
          - 요청 한도 초과(429) 시 지수 백오프 재시도 
        """
        raise NotImplementedError()
    
    def _fetch_synthetic(
        self,
        ticker: str, 
        start_date: date, 
        end_date: date,
    ) -> tuple[str, list[Bar]]:
        """ Pykrx 라이브러리를 이용해 OHLCV 생성 → 분봉 갯 수 = 390 * 영업일 수 """
        start_date_str = start_date.strftime(cfg.run.date_format)
        end_date_str = end_date.strftime(cfg.run.date_format)
        # pykrx 일봉 조회 라이브러리:
        # ─ 컬럼 = 시가, 고가, 저가, 종가, 거래량 (한국어 컬럼명)
        df = krx.get_market_ohlcv_by_date(start_date_str, end_date_str, ticker)
        print(msg.data.fetch.pykrx_info(start_date_str, end_date_str, ticker, df))
        if df.empty:
            return cfg.store.load_pykrx, []
        
        bars: list[Bar] = []
        rng = np.random.default_rng(seed=cfg.run.seed)
        for row in df.itertuples():
            d: date = cast(pd.Timestamp, row.Index).date()
            ohlcv: dict[str, Any] = {
                "open": float(cast(Any, row.시가)),
                "high": float(cast(Any, row.고가)),
                "low": float(cast(Any, row.저가)),
                "close": float(cast(Any, row.종가)),
                "volumn": int(cast(Any, row.거래량)),
            }
            bars.extend(_synthesize_minute_bars(ticker, d, ohlcv, rng))

        return cfg.store.load_pykrx, bars
    
def _synthesize_minute_bars(
    ticker: str,
    curr_date: date,
    ohlcv: dict[str, Any],
    rng: np.random.Generator
) -> list[Bar]:
    """ 일봉 OHLCV을 이용해 분봉 390개 합성 (브라운 운동 기반). """
    _open, _high, _low, _close, _volumn = \
        ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volumn"]
    
    num = cfg.run.minutes_per_day

    # 분봉별 드리프트와 변동성
    drift = (_close / _open - 1.0) / num 
    vol = abs(_high - _low) / _open / np.sqrt(num) * 1.5
    increments = rng.normal(drift, vol, num)

    # 누적 수익률 → 가격 경로
    prices = _open * np.cumprod(1 + increments)
    p_min, p_max = prices.min(), prices.max()
    

        
