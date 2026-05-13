""" HistoricalDataLoader
    과거 분봉 데이터 수집 및 로컬 캐싱

    [수집 전략]
    - 1순위: LocalParquetStore 캐시 → 같은 구간이 저정되어 있으면 재수집 생략.
    - 2순위: KIS_APP_KEY 환경변수 설정 시 KIS_REST_API로 실제 분봉 조회
    - 3순위: pykrx 일봉 기반 합성 분봉 생성 (개발·테스트 전용)

    [합성 분봉의 한계]
    - 브라운 운동으로 생성한 분봉은 실제 틱 데이터와 통계 분포가 완전히 다르기 때문에,
      이 데이터로 나온 백테스트 결과는 전략 검증 수준이며, 실거래 시 반드시 KIS 분봉으로 교체
"""

from __future__ import annotations

from datetime import date, datetime, timedelta 
import numpy as np
import pandas as pd

from mps.data.types import Bar
from mps.data.store import LocalParquetStore
from mps.data.calendar import market_open_dt
from mps.sys.config import settings 
from mps.sys import constants as const
from mps.sys import messages as msg


class HistoricalDataLoader:

    def __init__(self, store: LocalParquetStore | None = None) -> None:
        self._store = store or LocalParquetStore()

    def load(
        self,
        ticker: str,
        start_date: date, 
        end_date: date, 
        force_refresh: bool = False,    # 강제 재수집 여부
    ) -> list[Bar]:
        """ 캐시 우선 로드, 단 캐시가 없거나 force_refresh=True이면 재수집 """
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=const.KST)
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=const.KST)
        print(f"  = data load dt: {start_dt} - {end_dt}")

        cached_data = self._store.load_bars(ticker, start_dt, end_dt)
        print(f"  = 읽어온 캐시 데이터 수: {len(cached_data):,}봉")
        if cached_data and not force_refresh:
            print(f"  = 캐시된 데이터를 리턴")
            return cached_data
        
        load_bars = self._fetch(ticker, start_date, end_date)
        self._store.save_bars(load_bars)
        print("  = KIS 데이터 호출 후 리턴")
        return load_bars 
    
    def _fetch(self, ticker: str, start: date, end: date) -> list[Bar]:
        if settings.kis_app_key:
            return self._fetch_kis(ticker, start, end)
        return self._fetch_synthetic(ticker, start, end)
    
    def _fetch_kis(self, ticker: str, start: date, end: date) -> list[Bar]:
        """ KIS REST API 분봉 조회 (stub: API 키 설정 후 kis_client.py에서 구현 예정) 

            구현 시 고려사항:
            · KIS API는 한 번에 최대 200봉만 반환 → 날짜 범위를 쪼개서 반복 호출
            · OAuth 토큰 만료(약 24시간) 처리 필요
            · 요청 한도 초과(429) 시 지수 백오프 재시도
        """
        raise NotImplementedError(msg.NOT_IMPLEMENTED_KIS_FETCH)
    
    def _fetch_synthetic(self, ticker: str, start: datetime, end: datetime) -> list[Bar]:
        """ pykrx 일봉 OHLCV → 분봉 390개(1일) X 영업일 수 합성
        
            [경고]
            · 개발·테스트 전용. 실거래 배포 시 반드시 실제 분봉으로 교체해야 함
        """
        try:
            from pykrx import stock as krx 
        except ImportError:
            raise RuntimeError(msg.NOT_INSTALL_PYKRX)
        
        print("  - 개발 및 테스트용 분봉 데이터 생성...")
        
        start_date_str = start.strftime(const.DATE_FORMAT)
        end_date_str = end.strftime(const.DATE_FORMAT)
        print(msg.GET_KRX_DATE(start_date_str, end_date_str))

        # pykrx 일봉 조회: 컬럼 = Open, High, Low, Close, Volume (한국어 컬럼명)
        ohlcv_df = krx.get_market_ohlcv_by_date(start_date_str, end_date_str, ticker)
        if ohlcv_df.empty:
            print(msg.GET_KRX_DATA_EMPTY)
            return []

        bars: list[Bar] = []
        # SEED 고정, 동일 조건에서 동일 합성 결과 재현 가능해야 함
        # rng: Random Number Generator (난수 생성기)
        rng = np.random.default_rng(seed=const.SEED)

        for day, row in ohlcv_df.iterrows():
            d = day.date() if hasattr(day, "date") else day 
            o, h, l, c, v = (
                float(row["시가"]), float(row["고가"]), float(row["저가"]),
                float(row["종가"]), int(row["거래량"]),
            )
            # 각 거래일마다 390개 분봉을 생성하여 bars에 추가
            bars.extend(_synthesize_minute_bars(ticker, d, o, h, l, c, v, rng))

        return bars 
    

def _synthesize_minute_bars(
    ticker: str, work_date: date,
    day_open: float, day_high: float, day_low: float, day_close: float, day_volumn: int,
    rng: np.random.Generator
) -> list[Bar]:
    """ 일봉 OHLCV → 분봉 390개 합성 """
    minute_bar_count = const.MINUTES_PER_DAY  # 390
    
    # 분봉별 드리프트와 변동성
    drift = (day_close / day_open - 1.0) / minute_bar_count
    vol = abs(day_high - day_low) / day_open / np.sqrt(minute_bar_count) * 1.5
    increments = rng.normal(drift, vol, minute_bar_count)

    # 누적 수익률 → 가격 경로
    prices = day_open * np.cumprod(1 + increments)
    prices = np.clip(prices, day_low, day_high)
    # 마지막 봉 종가를 일봉 종가에 맞춤
    prices[-1] = day_close

    # 거래량 분산: 실제 KOSPI 패턴 모방 (시가·점심·마감 집중)
    vol_weight = _volume_weights(minute_bar_count, rng)
    volumes = (vol_weight * day_volumn).astype(int)

    open_datetime = market_open_dt(work_date)
    bars: list[Bar] = []
    prev = day_open 

    for i in range(minute_bar_count):
        ts = open_datetime + timedelta(minutes=i)
        _open = prev
        _close = prices[i]
        _high = min(max(_open, _close) * (1 + rng.uniform(0, vol * 0.5)), day_high)
        _low = max(min(_open, _close) * (1 - rng.uniform(0, vol * 0.5)), day_low)
        bars.append(Bar(
            ticker=ticker, timestamp=ts, 
            open=round(_open, 0), high=round(_high, 0), 
            low=round(_low, 0), close=round(_close, 0), 
            volume=int(volumes[i]),
            is_complete=True,
        ))
        
        prev = _close 
    
    return bars

def _volume_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """ KOSPI 거래량 패턴 모방: 시가·점심·마감에 집중 
        실제 KOSPI 거래량은 장 초반(09:00~09:30)과 장 마감(15:00~15:30)에 급증하고,
        점심 시간대(12:00~12:30)에도 소폭 증가하는 U자형 패턴이 있음
    """
    w = rng.uniform(0.5, 1.5, n)
    w[:30] *= 3.0
    w[120:150] *= 1.5
    w[360:] *= 2.5
    return w / w.sum()  # 합이 1이 되도록 정규화


