from __future__ import annotations 

import pytest 
from datetime import datetime, date
from pathlib import Path 

from mps.core.config import cfg
from mps.core.types import Bar, DataSource
from mps.core.libs import logger
from mps.data.io import LocalParquetStore
from mps.tests.data._helper_test_data import make_bars


@pytest.fixture 
def store(tmp_path: Path) -> LocalParquetStore:
    # tmp_path는 pytest 내장 fixture
    return LocalParquetStore(base_dir=tmp_path)


def _start_end(bars: list[Bar]) -> tuple[datetime, datetime]:
    timestamp = [bar.timestamp for bar in bars]
    return min(timestamp), max(timestamp)


class TestLocalParquetStore:
    
    def test_save_and_load_roundtrip(self, store):
        bars = make_bars(10)
        start, end = _start_end(bars)
        ticker = bars[0].ticker

        store.save_bars(bars)
        loaded_bars = store.load_bars(ticker, start, end)
        
        # 임의개의 바를 생성하고 저장 후 불러와서 갯 수 비교
        assert len(loaded_bars) == len(bars)
        # ticker값 비교
        assert all(bar.ticker == ticker for bar in loaded_bars)
        # ohlcv 값 비교
        assert (
            [bar.close for bar in bars] == \
                pytest.approx([bar.close for bar in loaded_bars], abs=1.0)
        )
        
    def test_store_file(self, store):
        start = datetime(2025, 1, 1, tzinfo=cfg.sys.timezone)
        end = datetime(2025, 1, 31, tzinfo=cfg.sys.timezone)
        
        # 존재하지 않는 종목은 빈 값을 리턴함
        ticker = "not-ticker"
        result_bars = store.load_bars(ticker, start, end)
        assert result_bars == []
        
        # 리스트 경로
        bars = make_bars(10)
        # 유효한 데이터가 있는 경우에는 파일에 저장
        store.save_bars(bars)
        assert list(store._base_dir.iterdir()) != []
        
        # 같은 timestamp인 경우 마지막 들어온 데이터 저장
        v1_bars = make_bars(3, seed=1)
        v2_bars = make_bars(3, seed=2)
        
        v1_close = v1_bars[0].close 
        v2_close = v2_bars[0].close 
        
        store.save_bars(v1_bars)
        store.save_bars(v2_bars)
        
        ticker = v1_bars[0].ticker 
        start, end = _start_end(v1_bars)
        load_bars = store.load_bars(ticker, start, end)
        
        assert v1_close != load_bars[0].close   # 처음것과 같지 않고
        assert v2_close == load_bars[0].close   # 마지막 저장값과 같음
        
    def test_range_masking(self, store):
        bars = make_bars(10)
        store.save_bars(bars)
        # 처음 5개만 불러옴
        timestamp = sorted(bar.timestamp for bar in bars)
        load_bars = store.load_bars(bars[0].ticker, timestamp[0], timestamp[4])
        assert len(load_bars) == 5
        
        
from unittest.mock import patch 

from mps.data.io import HistoricalDataLoader

START_DATE = date(2025, 1, 5)
END_DATE   = date(2025, 1, 5)


@pytest.fixture 
def loader(store: LocalParquetStore) -> HistoricalDataLoader:
    return HistoricalDataLoader(store=store)

@pytest.fixture
def stored_bars(store: LocalParquetStore) -> list[Bar]:
    # 봉 10개를 미리 저장 후 리턴
    bars = make_bars(10)
    store.save_bars(bars)
    return bars 

def _date_range(bars: list[Bar]) -> tuple[str, date, date]:
    """ bar의 실제 timestamp에서 날짜 범위를 추출 ─ hardcoded 날짜 불일치 방지 """
    dates = [bar.timestamp for bar in bars]
    return bars[0].ticker, min(dates), max(dates)

        
class TestHistoricalDataLoader:
    
    def test_cache_hit(self, loader, stored_bars):
        ticker, start_date, end_date = _date_range(stored_bars)
        source, bars = loader.load(
            ticker, start_date, end_date, force_refresh=False
        )
        assert source == cfg.str.store 
        assert len(bars) > 0
        assert all(isinstance(bar, Bar) for bar in bars)
        
        # 캐시에 데이터가 있으면 "_fetch"를 실행하지 않음을 테스트
        with patch.object(loader, "_fetch") as mock_fetch:
            loader.load(ticker, start_date, end_date, force_refresh=False)
            # 캐시가 단 한번도 실행되지 않았음을 확인
            mock_fetch.assert_not_called()
            
    def test_returns_tuple_of_two(self, loader, stored_bars):
        ticker, start_date, end_date = _date_range(stored_bars)
        result = loader.load(ticker, start_date, end_date)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        bars = result[1]
        assert isinstance(bars, list)
        assert all(isinstance(bar, Bar) for bar in bars)
        