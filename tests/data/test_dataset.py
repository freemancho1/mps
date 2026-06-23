""" TripleBarrierDataset 단위 테스트 """
from __future__ import annotations 

import math 
import numpy as np 
import torch 
import pytest 
from datetime import datetime, timezone 
from typing import get_args

from mps.core.config import cfg, msg
from mps.core.types import Bar, TrackType
from mps.data.features import TripleBarrierLabeler, TripleBarrierDataset
from mps.core.libs import logger 


def _make_bars(count: int, base_price: float = 10_000.0) -> list[Bar]:
    rng = np.random.default_rng(seed=cfg.sys.seed)
    bars: list[Bar] = []
    price = base_price 
    
    for idx in range(count):
        ret = rng.normal(0.0, 0.005)
        close = max(price * (1+ret), 1.0)
        high = close * (1 + abs(rng.normal(0, 0.002)))
        low = close * (1 - abs(rng.normal(0, 0.002)))
        open = price 
        volume = int(rng.integers(100, 10_000))
        bars.append(Bar(
            ticker="000000",
            timestamp=datetime(2025, 1, 2, 9, idx%60, tzinfo=cfg.sys.timezone),
            open=open, high=high, low=low, close=close, volume=volume,
            is_complete=True,
        ))
        price = close 
    
    return bars 

def _make_flat_bars(count: int, price: float = 10_000.0) -> list[Bar]:
    """ OHLC 전부 동일한(분산=0) 합성 봉 생성 """
    return [
        Bar(
            ticker="111111",
            timestamp=datetime(2025, 1, 2, 9, idx%60, tzinfo=cfg.sys.timezone),
            open=price, high=price, low=price, close=price, volume=1_000,
            is_complete=True
        )
        for idx in range(count)
    ]
    

LOOKBACK = 20
HORIZON = cfg.barrier.time_horizon
NUMERIC = cfg.modeling.numeric_track
PATTERN = cfg.modeling.pattern_track
N_FEATURES_NUMERIC = 14
N_FEATURES_PATTERN = 5

MIN_BARS = LOOKBACK + HORIZON + 5 


@pytest.fixture 
def labeler() -> TripleBarrierLabeler:
    return TripleBarrierLabeler()

@pytest.fixture 
def bars() -> list[Bar]:
    return _make_bars(MIN_BARS)


# ─────────────────────────────────────
#   내장함수: __len__ / __getitem__
# ─────────────────────────────────────

class TestLenAndGetitem:
    def test_len_matches_valid_range(self, bars):
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        expected = len(bars) - HORIZON - (LOOKBACK - 1)
        assert len(ds) == expected, "잘못된 길이로 만들어짐."
        
    def test_len_zero_when_insufficient_bars(self):
        """ 봉 수가 lookback + horizon 미만이면 유효 샘플 없음. """
        too_few = _make_bars(LOOKBACK + HORIZON - 1)
        ds = TripleBarrierDataset(too_few, track=NUMERIC, lookback=LOOKBACK)
        assert len(ds) == 0
        
    def test_getitem_returns_tensor_and_int(self, bars):
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        X, y = ds[0]
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, int)
        
    def test_getitem_label_is_valid_class(self, bars):
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        valid_classes = set(cfg.data.dir2idx.values())
        for i in range(len(ds)):
            _, y = ds[i]
            assert y in valid_classes 
            
    def test_getitem_all_samples_consistent(self, bars):
        """ X와 y의 샘플 수가 일치해야 함. """
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        assert ds._X.shape[0] == len(ds._y) == len(ds)
        

# ─────────────────────────────────────
#   NUMERIC Track (Z-score)
# ─────────────────────────────────────    

class TestNumericTrack:
    def test_X_window_shape(self, bars):
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        assert ds._X.shape == (len(ds), LOOKBACK, N_FEATURES_NUMERIC)
        
    def test_X_dtype_float32(self, bars):
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        assert ds._X.dtype == np.float32
        
    def test_zscore_mean_near_zero(self, bars):
        """ 각 윈도우의 피처 평균이 0(Z-score 정규화)에 가까워야 함. """
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        # shape: [n_samples, lookback, n_features]
        window_means = ds._X.mean(axis=1)
        assert np.allclose(window_means, 0.0, atol=1e-4)
        
    def test_zscore_std_near_one(self, bars):
        """ 각 윈도우의 피처 표준편차가 1에 가까워야 함 (상수 피처 제외) """
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        window_stds = ds._X.std(axis=1)
        # 분산이 0인 피처(상수)는 정규화 후 0이 되므로 제외
        non_const = window_stds > 1e-6
        if non_const.any():
            assert np.allclose(window_stds[non_const], 1.0, atol=1e-4)
            
    def test_tensor_shape_from_getitem(self, bars):
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        X, _ = ds[-1]
        assert X.shape == (LOOKBACK, N_FEATURES_NUMERIC)
        assert X.dtype == torch.float32 
        
    def test_empty_dataset_shape(self):
        """ 유효 샘플이 없을 경우 X.shape가 (0, lookback, n_features) """
        too_few_bars = _make_bars(LOOKBACK + HORIZON - 1)
        ds = TripleBarrierDataset(too_few_bars, track=NUMERIC, lookback=LOOKBACK)
        assert ds._X.shape == (0, LOOKBACK, N_FEATURES_NUMERIC)


# ─────────────────────────────────────
#   pattern 트랙 (min-max)
# ─────────────────────────────────────

def test_pattern_track(bars):
    ds = TripleBarrierDataset(bars, track=PATTERN, lookback=LOOKBACK)
    # 윈도우 크기 체크
    assert ds._X.shape == (len(ds), LOOKBACK, N_FEATURES_PATTERN)
    # 데이터 타입 체크
    assert ds._X.dtype == np.float32 
    # 모든 컬럼이 정규화 되었기 때문에 0~1 사이에 있어야 함.
    assert ds._X.min() >= -cfg.sys.zero 
    assert ds._X.max() <= 1.0 + cfg.sys.zero 
    
    # 가격이 0일때 NaN이나 inf가 아닌 0으로 채워져야 함
    flat_bars = _make_flat_bars(MIN_BARS)
    flat_ds = TripleBarrierDataset(flat_bars, track=PATTERN, lookback=LOOKBACK)
    assert np.isfinite(flat_ds._X).all()
    # 가격만 뽑아서 처리
    prices_window = flat_ds._X[:, :, :4]
    assert np.allclose(prices_window, 0.0)
    
    # 값이 없는 데이터셋 크기 (0, lookback, n_features)
    too_few_bars = _make_bars(LOOKBACK + HORIZON - 1)
    few_ds = TripleBarrierDataset(too_few_bars, track=PATTERN, lookback=LOOKBACK)
    assert few_ds._X.shape == (len(few_ds), LOOKBACK, N_FEATURES_PATTERN)   # len(few_ds) = 0
    
    # Tensor shape
    x, _ = ds[0]
    assert x.shape == (LOOKBACK, N_FEATURES_PATTERN)
    

# ─────────────────────────────────────
#   class_counts
# ─────────────────────────────────────

class TestClassCounts:

    def test_class_count(self, bars):
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK)
        
        # 클래스 카운트 전체 합 = 전체 데이터셋 크기
        assert ds.class_counts().sum() == len(ds)

        # 클래스 갯 수 확인
        assert len(ds.class_counts()) == cfg.lstm.num_classes

        # 각 클래스 갯 수가 최소 0개 이상
        assert (ds.class_counts() >= 0).all()


# ─────────────────────────────────────
#   유효성 검사
# ─────────────────────────────────────

class TestValidation:

    def test_invalid_track_raises(self, bars):
        with pytest.raises(TypeError) as exc_info:
            # 실제 코드에서 TypeError가 발생하지 않으면 여기서 오류가 발생함.
            TripleBarrierDataset(bars, track="invalid_track", lookback=LOOKBACK)
        # logger.error(str(exc_info.value))

    def test_custom_labeler_accepted(self, bars):
        time_horizon = 30
        labeler = TripleBarrierLabeler(0.03, 0.015, time_horizon=time_horizon)
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=LOOKBACK, labeler=labeler)
        # cfg에 time_horizon이 있더라도, labeler에 들어온 time_horizon에 의해 ds가 만들어짐
        expected = len(bars) - time_horizon - (LOOKBACK - 1)
        logger.debug(f"데이터셋 크기: {len(ds)}{ds._X.shape}")
        assert len(ds) == expected 

    def test_custom_lookback(self, bars):
        custom_lookback = 10
        ds = TripleBarrierDataset(bars, track=NUMERIC, lookback=custom_lookback)
        expected = len(bars) - HORIZON - (custom_lookback - 1)
        logger.debug(f"데이터셋 크기: {len(ds)}{ds._X.shape}")
        assert len(ds) == expected 
        assert ds._X.shape == (len(ds), custom_lookback, N_FEATURES_NUMERIC)