""" 
HistoricalSimulator ─ 과거 분봉 재생 기반 백테스트 엔진

[전체 파이프라인 한눈에 보기]
  - Bar 리스트 → (BarValidator 통과) → 전체 deque 버퍼에 순차 추가
  - 매 봉마다:
    1. 보유 포지션 체크 (StoplossTakeprofitGuard): 청산 조건 만족 시 청산
    2. 룩백 미달 봉은 건너뜀 (지표 계산 불가)
    3. 포지션 있으면 신규 신호 생략 (동시 다중 포시션 없음 ─ Phase-1 단순화)
    4. 피처 추출 + 정규화 → 두 트랙 모델 추론
    5. 신호 합의 + 지연 필터 + 점수 필터
    6. 수량 계산 → TripleBarrier 기준으로 주문 생성 → 체결
  - 전체 거래 기록 → PerformanceEvaluator → PerformanceReport 반환
  
[Phase-1 단순화 사항]
  - 단일 포지션: open_order 변수 하나로 관리 (동시 다중 포지션 없음)
  - 항상 시장가 체결
  - 공매도는 코드 상 지원하나 Phase-1 신호 필터링으로 사실상 미 발생
"""
from __future__ import annotations 

from typing import cast, Optional
from collections import deque 
from dataclasses import replace 

from mps.config import cfg, msg 
from mps.core.types import Bar, Order, TradeRecord, PerformanceReport, ExitReason
from mps.core.ports import NumericModelPort, PatternModelPort
from mps.pp.features.validator import BarValidator
from mps.pp.features.extractor import FeatureExtractor
from mps.pp.features.normalizer import NumericNormalizer, PatternNormalizer
from mps.models.factory import build_numeric_model, build_pattern_model
from mps.trade.signal import SignalAggregator, LatencyFilter, SignalFilter
from mps.trade.observability import SignalLogger, OrderLogger
from mps.trade.observability import LatencyMonitor
from mps.trade import PositionSizer, BuildOrder, PaperTrader, CostModel
from mps.trade import StopLossTakeProfitGuard
from .evaluator import PerformanceEvaluator


class HistoricalSimulator:
    def __init__(
        self,
        capital: Optional[float] = None,
        lookback_minutes: Optional[int] = None,
        numeric_model: Optional[NumericModelPort] = None, 
        pattern_model: Optional[PatternModelPort] = None,
    ) -> None:
        self._capital = capital or cfg.run.init_capital                         # 10,000,000.0원
        self._lookback_minutes = lookback_minutes or cfg.run.lookback_minutes   # 120
        
        self._validator = BarValidator()
        self._extractor = FeatureExtractor()
        self._numeric_normalizer = NumericNormalizer()
        self._pattern_normalizer = PatternNormalizer()
        self._numeric_model = numeric_model or build_numeric_model()
        self._pattern_model = pattern_model or build_pattern_model()
        self._aggregator = SignalAggregator()
        self._latency_filter = LatencyFilter()
        self._signal_filter = SignalFilter()
        self._sizer = PositionSizer(capital=self._capital)
        self._build_order = BuildOrder()
        self._trader = PaperTrader()
        self._cost_model = CostModel()
        self._sltp_guard = StopLossTakeProfitGuard()

        self._signal_logger = SignalLogger()
        self._order_logger = OrderLogger()
        self._latency = LatencyMonitor()
        
    def run(self, bars: list[Bar]) -> PerformanceReport:
        print(msg.trade.bt.sim_info(bars))
        
        # look-ahead bias 발생 위험 제거를 위해 is_complete=False 봉 제거
        bars = self._validator.filter(bars)
        
        # 윈도우 크기가 룩백+1 이상이 안되면 의미있는 백테스트가 불가하기 때문에 스킵
        # 12일(테스트 일자(10일) + 버퍼(2일)) * 390봉 = 4680봉인데, 
        #   이중에 필터링이 아무리 많이 당해봐야 121(lookback_minutes + 1)봉 보다는 많치 않을까? 싶네...
        if len(bars) < self._lookback_minutes + 1:
            raise ValueError(msg.trade.bt.sim_skip_err(bars, self._lookback_minutes + 1))
        
        # 상태변수 초기화
        # maxlen = lookback + 50 → 가장 오래된 봉은 deque 특성상 자동 삭제됨
        #                   + 50 = 기술지표 초기화 구간(NaN봉)을 여유롭게 포함하기 위함
        buffer: deque[Bar] = deque(maxlen=self._lookback_minutes + 50)  # 120 + 50 = 170
        trades: list[TradeRecord] = []      # 완결된 거래에 대한 기록
        cash = self._capital                # 현재(독립된) 시뮬레이터에서 사용할 수 있는 초기 자본금
        open_order: Optional[Order] = None  # 현재 보유 중인 포지션(거래 상태, None = 거래 없음)
        
        # ── 메일 루프: 봉 하나씩 생성 ─────────────────────
        for bar in bars:
            buffer.append(bar)
            
            # ── 1. 현재 보유중인 포지션이 있으면 청산 체크 ───────────
            # open_order(매수한 주식)가 있으면 현재 봉의 고가·저가로,
            # TakeProfit, StopLoss, 만료 조건(60분 경과 or 종료 15분 전) 확인
            if open_order is not None:
                action = self._sltp_guard.check(open_order, bar)

                if action != cfg.str.hold:
                    # 진입가(price)는 진입 시 체결가로 항상 채워짐(아래 self._trader 이후에 추가됨)
                    # → 아래 비용·손익 계산보다 먼저 Optional[float]에서 None을 뺀 float으로 변경
                    assert open_order.price is not None     # → pylance error 처리용

                    # 청산 기준가:
                    #  - 장벽 도달(TAKE_PROFIT·STOP_LOSS) → 해당 장벽 가격에서 체결 가정
                    #  - 시간·강제 청산(TIME_OUT·FORCE_CLOSE) → 시장가(현재 봉 종가)
                    exit_ref_price = self._exit_reference_price(open_order, action, bar)
                    # 청산의 진입은 "반대 방향" 주문 → PaperTrader 슬리피지도 반대로 적용
                    #  - 롱 청산 = 매도는 더 싸게, 숏 청산 = 매수는 더 비싸게 체결되어야 보수적임.
                    exit_side = cfg.key.SELL \
                        if open_order.direction == cfg.key.BUY else cfg.key.BUY
                    # open_order을 복사해 direction을 변경 후 복사본을 리턴함
                    exit_order = replace(open_order, direction=exit_side)
                    result = self._trader.submit_order(exit_order, exit_ref_price)

                    # 현금성 수수료(슬리피지는 체결가에 이미 반영됨)
                    sell_fee = self._cost_model.sell_cost(result.filled_price, result.filled_quantity)
                    buy_fee = self._cost_model.buy_cost(open_order.price, open_order.quantity)
                    roundtrip_fee = buy_fee + sell_fee 
                    self._order_logger.log(exit_order, result)

                    # 손익 계산: (진입 체결가 vs 청산 체결가 차이) * 수량
                    if open_order.direction == cfg.key.BUY:
                        # BUY 진입 → 청산가가 높을수록 이익
                        pnl = (result.filled_price - open_order.price) * result.filled_quantity
                    else:
                        # SELL 진입 → 청산가가 낮을수록 이익 (공매도?)
                        pnl = (open_order.price - result.filled_price) * result.filled_quantity

                    # 현금 복원: 진입에 묶인 금액 + 손익 - 청산 수수료
                    # ─ 진입 시 buy_fee는 이미 차감됐고, 진입가·손익이 모두 체결가 기준이라 정합
                    cash += open_order.price * open_order.quantity + pnl - sell_fee 

                    # 거래 기록 생성 (진입 + 청산 쌍, 비용은 왕복 수수료)
                    trades.append(TradeRecord(
                        ticker=open_order.ticker,
                        direction=open_order.direction,
                        entry_price=open_order.price,
                        exit_price=result.filled_price,
                        quantity=open_order.quantity,
                        entry_time=open_order.order_id,
                        exit_time=bar.timestamp,
                        exit_reason=cast(ExitReason, action),
                        cost=roundtrip_fee
                    ))
                    # 포지션 해제
                    open_order = None
            
            # ── 2. 룩백 미달 구간은 신호생성 생략 ───────────────
            if len(buffer) < self._lookback_minutes:
                continue
            
            # ── 3. 현재 미처리 포지션이 있으면 신규 구매 생략 ──────────
            if open_order is not None:
                continue 
            
            # ── 4. 신규 구매 절차 ───────────────────────
            buffer_list: list[Bar] = list(buffer)
            
            # 4-1. 피처 추출 및 정규화 --------------
            with self._latency.measure(cfg.key.feature):
                raw = self._extractor.extract(buffer_list)
                numeric_input = self._numeric_normalizer.transform(buffer_list, raw)
                pattern_input = self._pattern_normalizer.transform(buffer_list)

            # 4-2. 수치·패턴 트랙 신호 생성 ------------
            with self._latency.measure(cfg.key.numeric):
                numeric_signal = self._numeric_model.run(numeric_input)
            with self._latency.measure(cfg.key.pattern):
                pattern_signal = self._pattern_model.run(pattern_input, buffer_list)

            # 4-3. 신호 합의 + 가드 적용 --------------
            trade_signal = self._aggregator.combine(numeric_signal, pattern_signal)
            trade_signal = self._latency_filter.filter(trade_signal)
            if not (trade_signal := self._signal_filter.filter(trade_signal)):
                continue 

            # 4-4. 최종 통과한 신호 기록
            self._signal_logger.log(trade_signal)

            # ── 5. 수량 계산 ────────────────────────
            # 가격과 1회 투자 금액을 이용해 구매할 주식 수량 계산
            # 1회 투자 금액 = min(현재 금액, 초기 자본 * 10%)
            if (quantity := self._sizer.calc_quantity(bar.close, cash)) <= 0:   # 현금 부족
                continue

            # ── 6. 주문 생성 및 체결 ────────────────────
            # OrderBuilder: TakeProfit·StopLoss 가격 + 만료 시간 계산 → Order 객체 생성
            order = self._build_order.build(trade_signal, bar, quantity)
            # PaperTrader: 시장가 즉시 체결
            result = self._trader.submit_order(order, bar.close)
            # 진입가 = 실제 체결가(슬리피지 포함)로 기록
            # → 청산 손익·현금 원장·평가 수익률이 모두 동일 기준(체결가)을 쓰도록 정합
            order.price = result.filled_price

            # 현금성 수수료 (슬리피지는 체결가에 이미 반영 ─ 이중계상 없음)
            buy_fee = self._cost_model.buy_cost(
                result.filled_price, result.filled_quantity
            )
            # 현금 차감: 체결금액 + 수수료
            cash -= (result.filled_price * result.filled_quantity) + buy_fee
            self._order_logger.log(order, result)
            # 매수한 내용이 있으니 포지션 보유 시작
            open_order = order

        print(msg.trade.bt.sim_result(self._latency.summary()))
        # ── 성과 평가 ───────────────────────────
        evaluator = PerformanceEvaluator()
        report = evaluator.evaluate(trades, self._capital)
        return report
    
    @staticmethod
    def _exit_reference_price(order: Order, action: str, bar: Bar) -> float:
        """ 
        청산 액션별 기준 체결가 결정.
          - TAKE_PROFIT → 익절 장벽 가격
          - STOP_LOSS   → 손절 장벽 가격
          - TIME_OUT / FORCE_CLOSE → 시장가 (현재 봉 종가)

        실제 체결가는 PaperTrader가 이 기준가에 슬리피지를 반영해 산출
        """
        if action == cfg.str.take_profit:
            return order.take_profit
        if action == cfg.str.stop_loss:
            return order.stop_loss
        return bar.close
    
