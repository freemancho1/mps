from __future__ import annotations 

import numpy as np 
import pytest 

from mps.core.config import cfg 
from mps.data.features import TripleBarrierLabeler
from mps.data.features import FeatureExtractor, BarValidator
from mps.data.features import NumericNormalizer, PatternNormalizer
from mps.tests.data._helper_test_data import make_bars


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
        
        
class TestNormalizer:
    # TODO 9999-9999: 여기 