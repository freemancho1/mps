from __future__ import annotations 

import numpy as np 
import pytest 

from mps.core.config import cfg 
from mps.core.libs import logger 
from mps.core.types import NumericInput, PatternInput
from mps.data.features import TripleBarrierLabeler
from mps.data.features import FeatureExtractor, BarValidator
from mps.data.features import NumericNormalizer, PatternNormalizer
from mps.tests.data._helper_test_data import make_bars
from tests.data._helper_test_data import make_flat_bars


WINDOW = 30
WARMUP = 50


@pytest.fixture 
def bars():
    return make_bars(WINDOW + WARMUP)


class TestFeatureExtactorShape:
    
    def test_result_value(self, bars):
        result = FeatureExtractor.extract(bars)
        
        # shape
        assert result.shape == (len(bars), cfg.modeling.feature_count)
        
        # type
        assert result.dtype == np.float32 
        
        # NaN 처리 완료 여부
        assert np.isfinite(result).all()
        
        # 20개 미만 봉으로도 정상적으로 작동(ret_20에서 rolling(20)을 하는데...)
        # rsi·atr등 window가 14개가 필요한 지표가 있어 최소 봉 갯 수는 14개임.
        bars = make_bars(14)
        result = FeatureExtractor.extract(bars)
        assert result.shape == (len(bars), cfg.modeling.feature_count)
        assert np.isfinite(result).all()
        
        # 컬럼 갯 수 = 속성 갯 수 = 14
        assert result.shape[1] == len(cfg.modeling.feature_names)
        

class TestBarValidator:
    
    def test_validator(self):
        bars = make_bars(10)
        bars[3].is_complete = bars[7].is_complete = False 
        result = BarValidator.filter(bars)
        assert len(result) == 8
        assert all(bar.is_complete for bar in result)
        
        # 빈 배열에 대해서는 배열로 리턴
        assert BarValidator.filter([]) == []
        
        # 모드 리셋하고 확인하기
        for bar in bars:
            bar.is_complete = False 
        assert BarValidator.filter(bars) == []
        
        # 하나 임의 제거
        bars = make_bars(3)
        bars[1].is_complete = False 
        result = BarValidator.filter(bars)
        assert result == [bars[0], bars[2]]
        
        # validator 점검
        assert BarValidator.validator(bars[0]) is True 
        assert BarValidator.validator(bars[1]) is False
        
        
class TestTripleBarrierLabeler:
    
    def test_labeler(self):
        bars = make_bars(200)
        labels = TripleBarrierLabeler().label(bars)
        # 모든 봉에 대해서 라벨은 만듬.
        assert len(labels) == len(bars)
        
        valid = set(cfg.data.dir2idx.values())
        assert set(np.unique(labels)).issubset(valid)
        
        hold_idx = cfg.data.dir2idx[cfg.str.hold]
        for n in (0, 1):
            bars = make_bars(n) if n > 0 else []
            labels = TripleBarrierLabeler().label(bars)
            assert (labels == hold_idx).all()
            
        # 마지막 봉은 진입 다음 봉이 없어 항상 HOLD
        bars = make_bars(200)
        labels = TripleBarrierLabeler().label(bars)
        assert labels[-1] == hold_idx 
        
        # 각 라벨값 갯 수의 합은 전체 봉 갯 수
        labeler = TripleBarrierLabeler()
        labels = labeler.label(bars)
        dist = labeler.label_distribution(labels)
        assert sum(dist.values()) == len(bars)
        
        
WINDOW = 20


@pytest.fixture 
def bars():
    return make_bars(WINDOW + 10)

@pytest.fixture 
def feature_matrix(bars):
    return FeatureExtractor.extract(bars)

class TestNumericNormalizer:
    
    def test_numeric_normalizer(self):
        # matrix는 bars의 갯 수 만큼 만들어지고,
        bars = make_bars(WINDOW - 1)
        matrix = FeatureExtractor.extract(bars)
        # 매트릭스가 윈도우 크기(룩백)보다 적으면 오류 발생
        with pytest.raises(ValueError) as exe_info:
            NumericNormalizer(window_size=WINDOW).transform(bars, matrix)
        logger.test(str(exe_info.value))
        
        # 함수 리턴값이 NumericInput 타입인지 확인
        bars = make_bars(WINDOW + 10)
        matrix = FeatureExtractor.extract(bars)
        result = NumericNormalizer(window_size=WINDOW).transform(bars, matrix)
        assert isinstance(result, NumericInput)
        
        # 결과값이 입력값 마지막 봉 기준 여부 판단
        assert result.ticker == bars[-1].ticker 
        assert result.timestamp == bars[-1].timestamp 
        logger.point(f"ticker = {result.ticker}, timestamp = {result.timestamp}")
        
        # window, raw_window shape와 window_size의 값 확인
        assert result.window.shape == (WINDOW, cfg.modeling.feature_count)
        assert result.raw_window.shape == (WINDOW, cfg.modeling.feature_count)
        logger.point(f"window.shape = {result.window.shape}")
        logger.point(f"raw_window.shape = {result.raw_window.shape}")
        
        # 0에 가까운 평균, 1에 가까운 분산으로 만들어진 Z-score 정규화
        means = result.window.mean(axis=0)
        stds = result.window.std(axis=0)
        non_const = stds > 1e-6
        assert np.allclose(means, 0.0, atol=1e-4)
        if non_const.any():
            assert np.allclose(stds[non_const], 1.0, atol=1e-4)
        logger.point(f"means sum: {means.sum()}")
        logger.point(f"stds: {stds.sum() / len(stds)}, min stds: {stds.min()}")
        # logger.point(f"stds: {stds}")
        
        # window_size: 입력값(없으면 설정의 룩백값)
        assert result.window_size == WINDOW 
        logger.point(f"window size: {result.window_size}")
        
    
class TestPatternNormalizer:
    
    def test_too_few_bars(self):
        bars = make_bars(WINDOW - 1)
        with pytest.raises(ValueError) as exe_info:
            PatternNormalizer(window_size=WINDOW).transform(bars)
        logger.test(f"봉 갯 수 에러: {str(exe_info.value)}")

    def test_return_pattern_input(self, bars):
        result = PatternNormalizer(window_size=WINDOW).transform(bars)
        assert isinstance(result, PatternInput)
        logger.test(f"PatternNormalizer 리턴값 타입: {type(result)}")        
        
        base_bar = bars[-1]
        assert result.ticker == base_bar.ticker 
        assert result.timestamp == base_bar.timestamp 
        logger.test(f"결과 종목: {result.ticker}, 시간: {result.timestamp}")
        
        assert result.ohlcv_norm.shape == (WINDOW, 5)
        logger.test(f"결과 ohlcv shape: {result.ohlcv_norm.shape}")
        
        assert result.ohlcv_norm.dtype == np.float32 
        logger.test(f"결과 ohlcv data type: {type(result.ohlcv_norm)}")
        
        prices = result.ohlcv_norm[:, :4]
        assert prices.min() >= -1e-6
        assert prices.max() <= 1.0 + 1e-6
        logger.test(f"결과 ohlcv 가격 최소값: {prices.min()}, 최대값: {prices.max()}")
        
        volume = result.ohlcv_norm[:, 4]
        assert volume.min() >= -1e-6
        assert volume.max() <= 1.0 + 1e-6
        logger.test(f"결과 ohlcv 거래량 최소값: {volume.min()}, 최대값: {volume.max()}")
        
    def test_ticker_and_timestamp(self, bars):
        result = PatternNormalizer(window_size=WINDOW).transform(bars)
        base_bar = bars[-1]
        assert result.ticker == base_bar.ticker 
        assert result.timestamp == base_bar.timestamp 
        logger.test(f"결과 종목: {result.ticker}, 시간: {result.timestamp}")
        
    def test_flat_price_no_nan(self):
        """ 가격 변동이 없을 때 NaN 없이 0으로 값을 채워야 함 """
        flat_bars = make_flat_bars(WINDOW + 5)
        result = PatternNormalizer(window_size=WINDOW).transform(flat_bars)
        assert np.isfinite(result.ohlcv_norm).all()
        assert np.allclose(result.ohlcv_norm[:, :4], 0.0)
        logger.test(f"NaN 없이 정규화: {np.allclose(result.ohlcv_norm[:, :4], 0.0)}")
