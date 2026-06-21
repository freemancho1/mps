""" 
SignalPipeline ─ 봉 버퍼 → TradeSignal 변환의 단일 진입점

'신호 생성'과 '집행·장부 관리'를 분리해,
'봉 버퍼가 주어지면 신호(or None)를 돌려준다.'는 하나의 역할만 수행.
- 백테스트(HistoricalSimulator)와 실시간 엔진(LiveEngine, 예정)이 
  동일한 객체를 사용 → 백테스트에서 검증한 코드가 그대로 실거래로.
- 모델 주입(DI)으로 룰·학습 모델을 자유롭게 교체 
  (walk-forward 폴드별 새 모델도 파이프라인 새로 만들어 꽃으면 끝)

[처리 단계]
  - bars(버퍼)  → ① 피처 추출 → ② 트랙별 정규화 → ③ 두 트랙 추론
                → ④ 합의(Aggregator) → ⑤ 지연 필터 → ⑥ 점수 필터
                → TradeSignal or None
  - 각 단계의 소요 시간은 LatencyMonitor에 기록되며,
    통과한 신호는 SignalLogger가 기록함.
"""
from __future__ import annotations 

from typing import Optional 

from mps.config import cfg 
from mps.core.types import Bar, TradeSignal
from mps.core.ports import NumericModelPort, PatternModelPort
from mps.data.features import FeatureExtractor, NumericNormalizer, PatternNormalizer
from mps.model.factory import ModelFactory
from ._aggregator import SignalAggregator
from ._filter import LatencyFilter, ScoreFilter
from mps.trade.observabliity import SignalLogger, LatencyMonitor


class SignalPipeline:
    def __init__(
        self,
        numeric_model: Optional[NumericModelPort] = None,
        pattern_model: Optional[PatternModelPort] = None, 
        lookback_minutes: Optional[int] = None,
        latency_monitor: Optional[LatencyMonitor] = None,
    ) -> None:
        lookback = cfg.data.lookback_minutes \
            if lookback_minutes is None else lookback_minutes
        
        # ── 피처 단계
        self._extractor = FeatureExtractor()
        self._numeric_normalizer = NumericNormalizer(window_size=lookback)
        self._pattern_normalizer = PatternNormalizer(window_size=lookback)

        # ── 모델 주입
        model_factory = ModelFactory()
        self._numeric_model = model_factory.build_numeric() \
            if numeric_model is None else numeric_model
        self._pattern_model = model_factory.build_pattern() \
            if pattern_model is None else pattern_model
        
        # ── 합의·필터
        self._aggregator = SignalAggregator()
        self._latency_filter = LatencyFilter()
        self._score_filter = ScoreFilter()

        # ── 관측 가능성 
        self._signal_logger = SignalLogger()
        self._latency = LatencyMonitor() \
            if latency_monitor is None else latency_monitor

    @property 
    def latency(self) -> LatencyMonitor:
        """ 단계별 지연 통계 (백테스트 종료 시 summary() 출력용) """
        return self._latency
    
    def generate(self, bars: list[Bar]) -> Optional[TradeSignal]:
        """ 
        버퍼(최소 lookback봉)에서 신호 생성. 통과 못하면 None.

        호출 측(시뮬레이터/실시간 엔진)은 버퍼 충족·포지션 상태·진입 허용
        여부를 미리 검사해 불필요한 추론을 피하는 것이 좋음.
        """
        # ── ① + ② ─ 피처 추출·정규화
        with self._latency.measure(cfg.str.feature):
            raw = self._extractor.extract(bars)
            numeric_input = self._numeric_normalizer.transform(bars, raw)
            pattern_input = self._pattern_normalizer.transform(bars)

        # ── ③ ─ 두 트랙 독립 추론 (지연을 트랙별로 따로 측정)
        with self._latency.measure(cfg.str.numeric):
            numeric_signal = self._numeric_model.run(numeric_input)
        with self._latency.measure(cfg.str.pattern):
            pattern_signal = self._pattern_model.run(pattern_input, bars)

        # ── ④ + ⑤ ─ 합의 → 지연 필터 → 점수 필터 (None 이면 체인 중단)
        trade_signal = self._aggregator.combine(numeric_signal, pattern_signal)
        trade_signal = self._latency_filter.filter(trade_signal)
        trade_signal = self._score_filter.filter(trade_signal)
        if trade_signal is None:
            return None 
        
        # 최종 통과 신호만 기록 (관측 가능성 ─ '왜 진입했는가?'의 근거)
        self._signal_logger.log(trade_signal)
        return trade_signal