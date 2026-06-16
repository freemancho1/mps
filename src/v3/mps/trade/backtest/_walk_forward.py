""" 
WalkForwardValidator ─ 거래일 단위 윈도우별 "학습→평가" (룩어헤드 차단)

[왜 윈도우마다 학습하는가? ─ 인샘플 누수 제거]
  - 전체 기간으로 한 번 학습한 모델을, 그 기간의 부분 구간에서 "평가"하면,
    테스트 구간이 이미 학습에 포함되어 성과가 부풀려짐(인샘플 누수)
  - 따라서, 각 폴드(fold)는 "테스트 구간 이전 데이터로만" 학습한 모델로 평가해야 함.
    (결정8. "거래일 단위 시간순 분할 + 위크포워드"의 정신을 백테스트 단계까지 유지)
    
[롤링 윈도우 방식]
  - train_days: 테스트 직전 N 거래일을 학습 구간으로 사용 (롤링; 시장 국면 적용)
  - test_days: 학습 직후 M 거래일을 평가 구간으로 사용
  - buffer_days: 테스트 첫 봉의 지표 계산을 위한 워밍업(학습 구간 말미에서 가져옴)
  - 매 폴드마다 test_days 만큼 앞으로 슬라이딩 → 각 거래일이 테스트에 한 번만 등장
  
[한 폴드의 데이터 경계]
  - config 설정값: train_days=30, test_days=10, buffer_days=2, buffer_bars=120+50
    │←          train_days              →│← buffer →│← test_days →│
    └──────── 학습(라벨·정규화) ──┴─ 워밍업 ─┴ 평가(진입 허용) ┘
                                            ↑ trade_start
  - 학습:   train 구간 봉으로 TripleBarrierDataset 구성 후 LSTM·CNN 학습
  - 위밍업: train 구간 말미 buffer_days → 지표 계산용 (진입 금지)
  - 평가:   test 구간에서만 신규 진입 (HistoricalSimulator의 trade_start로 격리)
  
[비용 주의]: 폴드마다 두 모델이 각각 학습하므로 느리다. 
  - cfg.train.hyper.epochs, 폴드 수로 시간 조절
"""
from __future__ import annotations 

from datetime import date 
from typing import Optional 

from mps.config import cfg, msg 
from mps.core.types import Bar, PerformanceReport
from mps.core.calendar import market_open_datetime
from mps.data.features import TripleBarrierLabeler, TripleBarrierDataset
from mps.model.numeric.lstm import LSTMNet, LSTMModel
from mps.model.pattern.cnn import CNN1DNet, CNN1DPatternModel
from mps.model.trainer import ModelTrainer
from mps.freelibs import logger


class WalkForwardValidator:
    """ 
    WalkForward ─ 윈도우를 슬라이딩하며 매 폴드에서 "학습→평가" 반복
      - 각 폴드는 독립 모델·시뮬레이터를 생성하므로 상태 오염이 없음.
    """
    def __init__(
        self,
        train_days: Optional[int] = None,
        test_days: Optional[int] = None, 
        capital: Optional[float] = None,
    ) -> None:
        self._train_days = cfg.run.train_days if train_days is None else train_days     # 30
        self._test_days = cfg.run.test_days if test_days is None else test_days         # 10
        self._capital = capital or cfg.run.init_capital
        logger.info(msg.bt.wf_info(self))
        
    def run(self, bars: list[Bar]) -> list[PerformanceReport]:
        """ 전체 bars에 걸쳐 롤링 윈도우 학습/평가를 수행하고 폴드별 성과 보고서 반환 """
        all_days = sorted({bar.timestamp.date() for bar in bars})
        num_days = len(all_days)
        
        # 일자 → 봉 인덱싱 (반복 조회 비용 절감)
        daily_bars: dict[date, list[Bar]] = {}
        for bar in bars:
            daily_bars.setdefault(bar.timestamp.date(), []).append(bar)
            
        reports: list[PerformanceReport] = []
        curr_fold = 0
        
        # 테스트 구간 시작 인덱스는 시험 일자만큼 슬라이딩 후 진행
        last_range = num_days - self._test_days + 1
        for start_test_idx in range(self._train_days, last_range, self._test_days):
            start_train_idx = start_test_idx - self._train_days
            end_test_idx = start_test_idx + self._test_days
            
            train_days = all_days[start_train_idx:start_test_idx]
            warmup_days = all_days[start_test_idx - cfg.data.buffer_days:start_test_idx]
            test_days = all_days[start_test_idx:end_test_idx]
            
            train_bars = WalkForwardValidator._collect(daily_bars, train_days)
            eval_bars = WalkForwardValidator._collect(daily_bars, warmup_days + test_days)
            
            # 테스트 첫 거래일 개장 시각
            trade_start_datetime = market_open_datetime(test_days[0])
            
            curr_fold += 1
            logger.info(msg.bt.wf_fold_info(curr_fold, train_days, warmup_days, test_days))
            
            try:
                numeric_model, pattern_model = WalkForwardValidator._train_models(train_bars)
                
                # TODO 0616-1444: HistoricalSimulator() 작업 후
            
            except ValueError as ve:
                # 학습 데이터(학습 샘플·룩백 미달) 부족 → 이 폴드만 skip (정상적으로 발생 가능)
                logger.warning(msg.bt.err.insufficient_data(ve, curr_fold))
                continue
            
        return reports
                
    @staticmethod
    def _collect(daily_bars: dict[date, list[Bar]], days: list[date]) -> list[Bar]:
        results: list[Bar] = []
        for day in days:
            results.extend(daily_bars.get(day, []))
        return results
    
    @staticmethod 
    def _train_models(train_bars: list[Bar]) -> tuple[LSTMModel, CNN1DPatternModel]:
        """ train 구간 봉으로 수치(LSTM)·패턴(CNN) 모델을 새로 학습해 추론 에댑터로 감쌈. """
        
        # [이후 답변]: 백테스트를 진행하면서 각 폴드별로 각각 모델을 학습할 필요가 있나?
        #              ─ 전체 데이터를 이용해 한번에 numeric과 pattern 모델을 학습하고
        #                 이 모델을 이용해 각 구간(폴드) 별로 시뮬레이션을 돌리는 방식이 좋지 않나?
        
        labeler = TripleBarrierLabeler()
        trainer = ModelTrainer()
        
        numeric_ds = TripleBarrierDataset(train_bars, cfg.model.numeric_track, labeler=labeler)
        pattern_ds = TripleBarrierDataset(train_bars, cfg.model.pattern_track, labeler=labeler)
        
        lstm_net, _ = trainer.train(LSTMNet(**cfg.train.lstm_settings.to_dict()), numeric_ds)
        # logger.debug(msg.bt.wf_fold_train_result(lstm_net))
        cnn_net, _ = trainer.train(CNN1DNet(**cfg.train.cnn_settings.to_dict()), pattern_ds)
        # logger.debug(msg.bt.wf_fold_train_result(cnn_net))
        
        # 학습된 가중치를 그대로 보유한 추론 어댑터 생성
        # (파일 저장 없이 메모리로 전달)
        numeric_model = LSTMModel(device=cfg.model.torch_device)
        numeric_model.model.load_state_dict(lstm_net.state_dict())
        numeric_model.model.eval()
        numeric_model._trained = True 
        
        pattern_model = CNN1DPatternModel(device=cfg.model.torch_device)
        pattern_model.model.load_state_dict(cnn_net.state_dict())
        pattern_model.model.eval()
        pattern_model._trained = True 
        
        return numeric_model, pattern_model
        
        