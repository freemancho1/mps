""" 
HistoricalSimulator ─ 과거 분봉 재생 기반 백테스트 엔진

[전체 파이프라인]
  - 4개 협력 객체로 분리
    SignalPipeline  : 버퍼 → TradeSignal (신호 생성 전담)
    RiskManager     : 신호 → Order | Reject (신호 승인·사이징)
    Portfolio       : 현금·포지션·일일 손익 원장
    ExitPolicyPort  : 보유 포지션의 청산 판정 (TripleBarrier + 브레이크이븐)
  - 시뮬레이터 고유 기능: 봉을 순서대로 흘리며 위 객체들을 올바른 순서로 호출
  - 실시간 엔진(LiveEngine, 예정): 데이터 소스만 WebSocket으로 변경하고 4개 객체 재사용

[봉당 처리 순서 ─ 순서 자체가 look-ahead 차단 규칙]
  1. 날짜 변경 감지 → 일일 손익 리셋 (Portfolio.on_bar_date)
  2. 직전 봉에서 확정된 신호(pending)가 있으면, "이번 봉 시가"에 체결
    - 신호는 봉 t의 종가까지 보고 만들어지므로,
      실체결 가능한 가장 빠른 가격은 t+1의 시가이며, 같은 봉 종가 체결은 look-ahead
    - 라벨러(TripleBarrierLabeler)도 open[t+1] 진입 기준 → 학습·평가 정합.
    - pending은 1봉 한정. 체결 실패(거부)해도 이월하지 않음.
      ─ 신호 근거 봉에서 멀어질수록 정보가치가 사라지기 때문
  3. 보유 보지션 청산 체크 ─ 이번 봉에 진입했어도 같은 봉의 고·저가로
     즉시 청산 가능(시가 체결 후 봉 내 움직임은 진입 이후의 일)
  4. 신호 생성 조건 검사 (룩백·단일 포지션·trade_start·진입 허용)
     → SignalPipeline.generate() → 통과 시 pending에 저장
  
[로직 단순화] 단일 포지션 · 시장가 체결 · 롱 온리
"""
from __future__ import annotations 

from typing import cast, Optional 
from datetime import datetime 
from collections import deque 

from mps.config import cfg, msg 
from mps.core.types import Bar, Order, TradeSignal, TradeRecord, PerformanceReport
from mps.core.types import Reject, ExitReason
from mps.core.ports import NumericModelPort, PatternModelPort, ExitPolicyPort
from mps.data.features import BarValidator
from mps.trade.execution import RiskManager, PaperTrader, CostModel
from mps.trade.observabliity import OrderLogger
from ._portfolio import Portfolio
from mps.freelibs import logger 


class HistoricalSimulator:
    def __init__(
        self, 
        capital: Optional[float] = None, 
        lookback_minutes: Optional[int] = None, 
        numeric_model: Optional[NumericModelPort] = None,
        pattern_model: Optional[PatternModelPort] = None,
        exit_policy: Optional[ExitPolicyPort] = None,
    ) -> None:
        # 초기 자본과 과거 신호 데이터는 0 일 수 없음
        self._capital = capital or cfg.run.init_capital     
        self._lookback = lookback_minutes or cfg.data.lookback_minutes
        
        self._validator = BarValidator()
        # self._pipeline = 
        # TODO 0619-1012: run 작업 후

        self._risk = RiskManager(capital=self._capital)
        # TODO 0619-2302: exit_policy 작업 후
        # self._exit_policy: ExitPolicyPort = (
        #     exit_policy if exit_policy is not None else TripleBa
        # )

        self._trader = PaperTrader()
        self._cost_model = CostModel()
        self._order_logger = OrderLogger()


    def run(
        self, 
        bars: list[Bar], 
        trade_start: Optional[datetime] = None
    ) -> PerformanceReport:
        """ 
        과거 분봉을 재생하며 백테스트 → PerformanceReport.
        
        trade_start: 신규 진입 허용 시각.
          - walk-forward에서 위밍업(버퍼) 구간 봉은 지표 계산에만 쓰고,
            테스트 구간에서만 실제 진입하도록 격리하느데 사용.
          - 워밍업 구간에서도 '보유 포지션 청산'은 정상 수행됨.
        """
        logger.info(msg.bt.sim_info(trade_start, bars))
        
        valid_bars = self._validator.filter(bars)
        if len(valid_bars) < self._lookback + 1:
            raise ValueError(msg.bt.err.sim_data_size(len(valid_bars), self._lookback))
        
        # 버퍼 = 룩백 + 워밍업. deque-maxlen으로 오래된 봉 자동 폐기
        buffer: deque[Bar] = deque(maxlen=cfg.data.buffer_bars)     # 120+50 = 170봉
        portfolio = Portfolio(capital=self._capital)
        # 다음 봉 시가 체결 대기 신호
        pending_signal = None 

        for bar in bars:
            buffer.append(bar)

            # ── 1. 일일 손익 리셋 
            portfolio.on_bar_date(bar.timestamp.date())

            # ── 2. Pending 신호를 이번 봉 시가에 체결 시도
            if pending_signal is not None:
                if not portfolio.has_position:
                    # TODO 0619-1731
                    self._try_fill_entry(pending_signal, bar, portfolio)

    
    # ── 내부 헬퍼함수
    def _try_fill_entry(self, signal: TradeSignal, bar: Bar, portfolio: Portfolio) -> None:
        """ 
        진입 체결 절차: 시가 + 슬리피지 체결 → RiskManager 승인 → 원장 반영.

        체결가 확정 순서가 중요.
          1. 시가 기준으로 PaperTrader가 슬리피지 반영 체결가 산출
          2. '실제 체결가'로 RiskManager가 사이징·장벽·만료 산출
        장벽(±2%/-1%)이 체결가 기준이어야 라벨·청산 가드와 가격 기준이 일치.
        """
        # 1. 체결가 산출(슬리피지 포함) ─ 매수는 기준가보다 불리하게 체결
        slippage = bar.open * cfg.trade.cost.slippage_rate 
        fill_price = round(bar.open + slippage, 0)

        # 2. 승인 (컷오프·일일한도·사이징·주문 생성)
        approved = self._risk.approve(
            signal, bar, fill_price, portfolio.cash, portfolio.day_pnl
        )
        if isinstance(approved, Reject):
            # 거부도 기록 ─ '왜 진입하지 않았는가?'의 관측 가능성
            logger.warning(msg.bt.err.entry_reject(approved))
            return 
        
        order: Order = approved 
        # 3. 정식 제출 (체결 결과 객체 생성·기록 ─ 시뮬레이션 시각 주입)
        # TODO 9999-9999: self._order_logger 구현 후
        # result = self.