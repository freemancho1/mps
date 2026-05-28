""" 
HistoricalSimulator ─ 과거 분봉 재생 기반 백테스트 엔진

[전체 파이프라인 한눈에 보기]
  - Bar 리스트 → (BarValidator) → deque 버퍼에 순차 추가
  - 매 봉마다:
    1. 보유 포지션 체크 (StoplossTakeProfitGuard): 청산 조건 만족 시 청산
    2. 룩백 미달 봉은 건너뜀 (지표 계산 불가)
    3. 포지션 있으면 신규 신호 생략 (동시 다중 포지션 없음 ─ Phase 1 단순화)
    4. 피처 추출 + 정규화 → 두 트랙 모델 추론 
    5. 신호 합의 + 지연 필터 + 점수 필터
    6. 수량 계산 → TripleBarrier 기준으로 주문 생성 → 체결
  - 전체 거래 기록 → PerformanceEvaluator → PerformanceReport 반환
  
[단순화 사항 (Phase-1 기준)]
  - 단일 포지션: open_order 변수 하나로 관리 (동시 다중 포지션 없음)
  - 항상 시장가 즉시 체결
  - 공매도는 코드 상 지원하나 Phase-1 신호 필터링으로 사실상 미 발생
"""
from __future__ import annotations 

from typing import Optional 
from collections import deque 
from dataclasses import replace

from mps.sys import cfg, msg
from mps.sys.core.types import Bar, Order
from mps.pipline.evaluator import PerformanceEvaluator, PerformanceReport, TradeRecord
from mps.pipline.features import BarValidator, NumericalNormalizer, PatternNormalizer
from mps.pipline.models.numeric import FeatureExtractor, ThresholdModel
from mps.pipline.models.pattern import RuleBasedPatternEngine
from mps.pipline.signal import SignalAggregator, LatencyGuard, SignalFilter
from mps.pipline.observability.risk import CostModel, PositionSizer
from mps.pipline.observability.risk import TripleBarrierGuard, StopLossTakeProfitGuard
from mps.pipline.execution import PaperTrader, OrderStateTracker
from mps.pipline.observability import LatencyMonitor, SignalLogger, OrderLogger


class HistoricalSimulator: 
    def __init__(
        self,
        capital: float = cfg.run.capital,                   # 10,000,000.0원
        lookback_minutes: int = cfg.sys.lookback_minutes    # 120
    ) -> None:
        self._capital = capital 
        self._lookback_minutes = lookback_minutes
        # print(msg.hs.init(self))
        
        self._validator = BarValidator()
        self._extractor = FeatureExtractor()
        self._numeric_normalizer = NumericalNormalizer()
        self._pattern_normalizer = PatternNormalizer()
        self._numeric_model = ThresholdModel()
        self._pattern_engine = RuleBasedPatternEngine()
        self._aggregator = SignalAggregator()
        self._latency_guard = LatencyGuard()
        self._signal_filter = SignalFilter()
        self._cost_model = CostModel()
        self._sizer = PositionSizer(capital=self._capital)
        self._barrier_guard = TripleBarrierGuard()
        self._sltp_guard = StopLossTakeProfitGuard()
        self._trader = PaperTrader()
        self._tracker = OrderStateTracker()
        self._signal_logger = SignalLogger()
        self._order_logger = OrderLogger()
        self._latency = LatencyMonitor()
        
    def run(self, bars: list[Bar]) -> PerformanceReport:
        print(msg.hs.run_info(bars))
        # is_complete=False 봉이 섞여 있으면 look-ahead bias 발생 위험
        bars = self._validator.filter(bars)

        # 룩백 + 1 이상 없으면 의미 있는 백테스트 불가
        if len(bars) < self._lookback_minutes + 1:
            raise ValueError(msg.hs.lookback_under_err(bars, self._lookback_minutes+1))
        
        # 상태변수 초기화
        # maxlen = lookback + 50: 가장 오래된 봉이 자동 삭제 → 메모리 효율
        # +50은 기술 지표 초기화 구간(NaN봉)을 여유롭게 포함하기 위함
        buffer: deque[Bar] = deque(maxlen=self._lookback_minutes + 50)  # 120+50 = 170
        # print(msg.hs.size_check(bars, buffer))
        trades: list[TradeRecord] = []      # 완결된 거래에 대한 기록
        cash = self._capital                # 현재 사용 가능한 현금 (총 자산)
        open_order: Optional[Order] = None  # 현재 보유 중인 포지션 (None = 미보유)
        
        # ── 메인 루프: 봉 하나씩 생성 ─────────────────────
        for bar in bars:
            buffer.append(bar)

            # ── 1. 현재 보유중인 포지션이 있으면 청산 체크 ───────────
            # open_order가 있으면 현재 봉의 고가/저가로 TP·SL·만료 조건 확인
            if open_order is not None:
                action = self._sltp_guard.check(
                    open_order, bar.high, bar.low, bar.timestamp
                )
                
                if action != "HOLD": 
                    # 청산 기준가:
                    #  - 장벽 도달(TAKE_PROFIT/STOP_LOSS) → 해당 장벽 가격에서 체결 가정
                    #  - 시간/강제 청산(TIMEOUT/FORCE_CLOSE) → 시장가(현재 봉 종가)
                    exit_ref_price = self._exit_reference_price(open_order, action, bar)
                    # 청산은 진입의 '반대 방향' 주문 → PaperTrader 슬리피지도 반대로 적용
                    #  (롱 청산=매도는 더 싸게, 숏 청산=매수는 더 비싸게 체결되어야 보수적)
                    exit_side = "SELL" if open_order.direction == "BUY" else "BUY"
                    exit_order = replace(open_order, direction=exit_side)
                    result = self._trader.submit_order(exit_order, exit_ref_price)
                    
                    # order.price는 진입 시 체결가로 채워짐 (Optional 타입 좁힘)
                    assert open_order.price is not None                     
                    
                    # 현금성 수수료 (슬리피지는 체결가에 이미 반영됨 ─ 이중계상 없음)
                    sell_fee = self._cost_model.sell_cost(result.filled_price, result.filled_quantity)
                    buy_fee = self._cost_model.buy_cost(open_order.price, open_order.quantity)
                    roundtrip_fee = buy_fee + sell_fee
                    self._order_logger.log(exit_order, result)
                    
                    # 손익 계산: 진입 체결가 vs 청산 체결가 차이 * 수량
                    if open_order.direction == "BUY":
                        # BUY 진입 → 청산가가 높을수록 이익
                        pnl = (result.filled_price - open_order.price) * result.filled_quantity
                    else:
                        # SELL 진입 → 청산가가 낮을수록 이익 (공매도)
                        pnl = (open_order.price - result.filled_price) * result.filled_quantity
                        
                    # 현금 복원: 진입에 묶인 금액 + 손익 - 청산 수수료
                    #  (진입 시 buy_fee는 이미 차감됐고, 진입가/손익이 모두 체결가 기준이라 정합)
                    cash += open_order.price * open_order.quantity + pnl - sell_fee
                    
                    # 거래 기록 생성 (진입 + 청산 쌍, 비용은 왕복 수수료)
                    trades.append(TradeRecord(
                        ticker=open_order.ticker,
                        direction=open_order.direction,
                        entry_price=open_order.price,
                        exit_price=result.filled_price,
                        quantity=open_order.quantity,
                        entry_time=open_order.order_id,     # 진입 시 order_id ⇒ 진입 시각 문자열
                        exit_time=bar.timestamp,
                        exit_reason=action,
                        cost=roundtrip_fee,
                    ))
                    open_order = None                       # 포지션 해제

            # ── 2. 룩백 미달 구간은 신호 생성 생략 ───────────────
            # buffer에 lookback 봉 이상 쌓이기 전까지는 지표 계산이 의미 없음
            if len(buffer) < self._lookback_minutes:
                continue 

            # ── 3. 현재 미처리 포지션이 있으면 신규 구매 생략(한번에 하나의 거래만) ──
            if open_order is not None:
                continue

            # deque 슬라이싱을 위해 list로 변환
            buffer_list = list(buffer)

            # ── 4. 피처 추출 및 정규화 ───────────────────────
            with self._latency.measure("feature"):
                raw = self._extractor.extract(buffer_list)
                numeric_input = self._numeric_normalizer.transform(buffer_list, raw)
                pattern_input = self._pattern_normalizer.transform(buffer_list)
                
            # ── 5. 수치·패턴 트랙 신호 생성 ────────────────────
            with self._latency.measure("numerical"):
                numeric_signal = self._numeric_model.run(numeric_input)
            with self._latency.measure("pattern"):
                pattern_signal = self._pattern_engine.run(pattern_input, buffer_list)
                
            # ── 6. 신호 합의 + 가드 적용 ─────────────────────
            trade_signal = self._aggregator.combine(numeric_signal, pattern_signal)
            trade_signal = self._latency_guard.filter(trade_signal)
            trade_signal = self._signal_filter.filter(trade_signal)
            
            # 신호가 없으면 다음 봉으로 이동
            if trade_signal is None: 
                continue 
            
            # 최종 통과한 신호는 기록 (logs/signals.jsonl)
            self._signal_logger.log(trade_signal)
            
            # ── 7. 수량 계산 ───────────────────────────
            # PositionSizer: min(현재 현금, 초기 자본 * 10%) // 현재가
            quantity = self._sizer.calc_quantity(bar.close, cash)
            if quantity <= 0:       # 현금 부족
                continue            # 주문 건너뜀
            
            # ── 8. 주문 생성 및 체결 ───────────────────────
            # TripleBarrierGuard: TP/SL 가격 + 만료 시간 계산 → Order 객체 생성
            order = self._barrier_guard.build_order(trade_signal, bar.close, quantity, bar.timestamp)
            # order_id: "{ticker}_{HHMMSS}" 형식으로 진입 시각 추적 가능
            order.order_id = f"{bar.ticker}_{bar.timestamp.strftime('%H%M%S')}"
            
            # PaperTrader: 시장가 즉시 체결 (슬리피지 포함)
            result = self._trader.submit_order(order, bar.close)
            # 진입가 = 실제 체결가(슬리피지 포함)로 기록
            #  → 청산 손익·현금 원장·평가기 수익률이 모두 동일 기준(체결가)을 쓰도록 정합
            order.price = result.filled_price
            # 현금성 수수료 (슬리피지는 체결가에 이미 반영 ─ 이중계상 없음)
            buy_fee = self._cost_model.buy_cost(result.filled_price, quantity)
            # 현금 차감: 체결 금액 + 매수 수수료
            cash -= result.filled_price * quantity + buy_fee
            self._order_logger.log(order, result)
            open_order = order  # 포지션 보유 시작

        # ── 성과 평가 ───────────────────────────────
        evaluator = PerformanceEvaluator()
        report = evaluator.evaluate(trades, self._capital)
        
        return report

    @staticmethod
    def _exit_reference_price(order: Order, action: str, bar: Bar) -> float:
        """ 
        청산 액션별 기준 체결가 결정.
          - TAKE_PROFIT → 익절 장벽 가격
          - STOP_LOSS   → 손절 장벽 가격
          - TIMEOUT / FORCE_CLOSE → 시장가(현재 봉 종가)
        (실제 체결가는 PaperTrader가 이 기준가에 슬리피지를 반영해 산출)
        """
        if action == "TAKE_PROFIT":
            return order.take_profit
        if action == "STOP_LOSS":
            return order.stop_loss
        return bar.close