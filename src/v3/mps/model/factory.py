""" 
모델 팩토리 ─ cfg.sys.phase에 따라 트랙별 모델을 선택함.

[Phase 전환]
  - phase == 1: [Numeric] ThresholdModel(룰), [Pattern] RuleBasedPatternEngine(룰)
  - phase >= 2: [Numeric] LSTMModel(학습), [Pattern] CNN1DPatternModel(학습)

[안전 풀백]
  - phase >= 2이지만 학습 가중치가 없으면 해당 트랙은 phase-1로 풀백함.
    (학습을 하지 않는 상태에서 무작위 초기 모델로 매매하는 사고 방지)
  - 풀백 시 경고를 출력해 관측 가능하게 구현
"""
from __future__ import annotations 

from typing import Optional 

from mps.config import cfg, msg 
from mps.core.ports import NumericModelPort, PatternModelPort 
from mps.model.numeric.threshold import ThresholdModel
from mps.model.pattern.rules import RuleBasedPatternEngine
from mps.freelibs import logger 


class ModelFactory:
    def __init__(self, phase: Optional[int] = None) -> None:
        self._phase = cfg.sys.phase if phase is None else phase 

    def build_numeric(self) -> NumericModelPort:
        if self._phase > 1:
            # 지연 임포트: torch 의존성을 Phase-1 경로에 강제하지 않음.
            from mps.model.numeric.lstm import LSTMModel

            if cfg.path.lstm_model_fpath.exists():
                model = LSTMModel(weights_path=cfg.path.lstm_model_fpath)
                if model.is_trained:
                    return model 
                
            logger.warning(msg.bt.err.model_phase(self._phase, cfg.str.numeric))

        return ThresholdModel()

    def build_pattern(self) -> PatternModelPort:
        if self._phase > 1:
            from mps.model.pattern.cnn import CNN1DPatternModel

            if cfg.path.cnn_model_fpath.exists():
                model = CNN1DPatternModel(weights_path=cfg.path.cnn_model_fpath)
                if model.is_trained:
                    return model 
                
            logger.warning(msg.bt.err.model_phase(self._phase, cfg.str.pattern))

        return RuleBasedPatternEngine()