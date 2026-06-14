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
from typing import Optional, cast

from mps.config import cfg, msg
from mps.core.types import Bar 
from mps.core.calendar import market_open_datetime
from mps.data.io import LocalParquetStore
from mps.core.libs import to_float, to_int
from mps.freelibs import logger 


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
        start_datetime = datetime.combine(
            start_date, datetime.min.time(), tzinfo=cfg.sys.timezone
        )
        end_datetime = datetime.combine(
            end_date, datetime.max.time(), tzinfo=cfg.sys.timezone
        )

        cached_bars = self._store.load_bars(ticker, start_datetime, end_datetime)
        if cached_bars and not force_refresh:
            return cfg.str.store, cached_bars
        
        load_from, bars = self._fetch(ticker, start_date, end_date)
        logger.debug(msg.pp.store.load_result(load_from, bars))
        self._store.save_bars(bars)
        return load_from, bars

    def _fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> tuple[str, list[Bar]]:
        """ 데이터 획득 ─ KIS 우선하고 없으면 합성 """
        if cfg.kis.app_key:
            return self._fetch_kis(ticker, start_date, end_date)
        return self._fetch_synthetic(ticker, start_date, end_date)
    
    def _fetch_kis(
        self,
        ticker: str,
        start_date: date, end_date: date,
    ) -> tuple[str, list[Bar]]:
        """ 
        KIS REST API 분봉 조회 (stub ─ API 키 설정 후 kis_client.py에서 구현)

        [구현 시 고려사항]
          - 1회 최대 100~200봉만 반환 함 → 날짜 범위 분할 반복 호출 필요
          - OAuth 토큰 만료(약 24h) 처리, HTTP 상태코드 429 시 지수 백오프 전략 필요
        """
        raise NotImplementedError()
    
    def _fetch_synthetic(
        self,
        ticker: str,
        start_date: date, end_date: date,
    ) -> tuple[str, list[Bar]]: 
        """ pykrx 일봉 → 분봉 390개/일 합성 (브라운 운동 기반) """

        # KIS가 연결이 안된 사항에만 사용하는 라이브러리로 의존성 격리 차원의 지연 임포트
        from pykrx import stock as krx 

        start_date_str = start_date.strftime(cfg.sys.date_format)
        end_date_str = end_date.strftime(cfg.sys.date_format)
        # pykrx 일봉: 컬럼 = 시가·고가·저가·종가·거래량 (한국의 컬럼명)
        df = krx.get_market_ohlcv_by_date(start_date_str, end_date_str, ticker)
        logger.debug(msg.pp.store.krx_synthetic_info(df))
        if df.empty:
            return cfg.str.pykrx, []
        
        bars: list[Bar] = []
        rng = np.random.default_rng(seed=cfg.sys.seed)
        for row in df.itertuples():
            curr_date: date = cast(pd.Timestamp, row.Index).date()
            ohlcv: dict[str, float | int] = {
                cfg.key.open    : to_float(row.시가),
                cfg.key.high    : to_float(row.고가),
                cfg.key.low     : to_float(row.저가),
                cfg.key.close   : to_float(row.종가),
                cfg.key.volume  : to_int(row.거래량),
            }
            bars.extend(_synthesize_minute_bars(ticker, curr_date, ohlcv, rng))

        return cfg.str.pykrx, bars 
    
def _synthesize_minute_bars(
    ticker: str, 
    curr_date: date,
    ohlcv: dict[str, float | int],
    rng: np.random.Generator
) -> list[Bar]:
    """ 일봉 OHLCV → 분봉 390개 합성 (GBM 경로 + 일봉 레이지 강제 정합) """
    day_open = ohlcv[cfg.key.open]
    day_high = ohlcv[cfg.key.high]
    day_low = ohlcv[cfg.key.low]
    day_close = ohlcv[cfg.key.close]
    day_volume = ohlcv[cfg.key.volume]

    bar_count = cfg.market.minutes_per_day

    # 분봉별 드리프트·변동성
    drift = (day_close / day_open - 1.0) / bar_count 
    vol = abs(day_high - day_low) / day_open / np.sqrt(bar_count) * 1.5
    increments = rng.normal(drift, vol, bar_count)

    # 누적 수익률 → 가격 경로 (일봉 고저 범위로 선형 재스케일)
    prices = day_open * np.cumprod(1 + increments)
    p_max, p_min = prices.max(), prices.min()
    if p_max > p_min:
        prices = day_low + (prices - p_min) / (p_max - p_min) * (day_high - day_low)
    else:
        prices[:] = float(day_open)
    prices[-1] = float(day_close)   # 마지막 분봉 종가는 일봉 종가

    # 거래량 분산: KOSPI W 자형 패턴 모방
    volume_weights = _volume_weights(bar_count, rng)
    volumes = (volume_weights * day_volume).astype(int)

    open_datetime = market_open_datetime(curr_date)
    bars: list[Bar] = []
    prev = day_open 
    for idx in range(bar_count):
        curr_datetime = open_datetime + timedelta(minutes=idx)
        open = prev 
        close = prices[idx]
        high = max(open, close) * (1 + rng.uniform(0, vol * cfg.data.min_max_noise))
        low = min(open, close) * (1 - rng.uniform(0, vol * cfg.data.min_max_noise))
        high = min(high, day_high)
        low = max(low, day_low)
        bars.append(Bar(
            ticker=ticker,
            timestamp=curr_datetime,
            open=round(open, 0),
            high=round(high, 0),
            low=round(low, 0),
            close=round(close, 0),
            volume=int(volumes[idx]),
            is_complete=True
        ))
        prev = close

    return bars

def _volume_weights(bar_count: int, rng: np.random.Generator) -> np.ndarray:
    """ KOSPI 거래량 패턴: 시가·점심·마감 집중형 (W형) """
    weights = rng.uniform(0.5, 1.5, bar_count)
    weights[:30] *= cfg.data.volume_weight_09       # 장 초반 30분: 2.6배
    weights[120:150] *= cfg.data.volume_weight_12   # 점심: 1.5배
    weights[360:] *= cfg.data.volume_weight_15      # 장 마감 30분: 2.3배
    return weights / weights.sum()                  # sum = 1이되는 정규화

