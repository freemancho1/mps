from __future__ import annotations 

import os 
import math 
from dataclasses import dataclass, field 
from pathlib import Path
from zoneinfo import ZoneInfo 
from datetime import time

from . import _constants as const 


@dataclass
class SysConfig:
    seed: int = 42 
    phase: int = 1

    zero: float = const.ZERO
    
    timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo(const.CURR_TIMEZONE))
    date_format: str = const.DATE_FORMAT

    # 마켓 정보
    _open = const.MARKET_OPEN_TIME
    _close = const.MARKET_CLOSE_TIME
    market_open_time: time = time(int(_open[:2]), int(_open[3:]))
    market_close_time: time = time(int(_close[:2]), int(_close[3:]))
    minutes_per_day: int = const.MINUTES_PER_DAY
    
    # 룩백 윈도우: 신호 생성 전 반드시 확보해야 할 과거 봉 수
    lookback_minutes: int = const.LOOKBACK_MINUTES
    buffer_days = math.ceil(lookback_minutes / minutes_per_day) + 1
    force_close_minutes_before: int = const.FORCE_CLOSE_MINUTES_BEFORE
    force_refresh: bool = const.FORCE_REFRESH

    rsi_oversold: float = const.RSI_OVERSOLD
    rsi_overbought: float = const.RSI_OVERBOUGHT
    rsi_closeover_base: float = const.RSI_CLOSEOVER_BASE
    
    # 결합 정보
    numeric_combined_weight: float = const.NUMERIC_COMBINED_WEIGHT
    pattern_combined_weight: float = const.PATTERN_COMBINED_WEIGHT
    
    max_latency_ms: float = const.MAX_LATENCY_MS
    min_combined_score: float = const.MIN_COMBINED_SCORE
    
    # 거래 건당 최대 거래 금액
    max_position_pct: float = const.MAX_POSITION_PCT    # 초기 자본의 10%


@dataclass
class RunBacktestConfig:
    tickers: list[str] = field(default_factory=lambda: const.TEST_TICKERS)

    start_date: str = const.TEST_START_DATE
    end_date: str = const.TEST_END_DATE    
    capital: float = const.TEST_CAPITAL
    test_days: int = const.TEST_DAYS

    @dataclass(frozen=True)
    class _key:
        ticker: str = "--ticker"
        start_date: str = "--start"
        end_date: str = "--end"
        capital: str = "--capital"
        test_days: str = "--test_days"
        
    key: _key = field(default_factory=_key)
    
    
@dataclass 
class KisApiConfig:
    
    @property 
    def app_key(self) -> str:
        return os.environ.get(const.VAR_KIS_APP_KEY, "")
    
    @property 
    def app_secret(self) -> str: 
        return os.environ.get(const.VAR_KIS_APP_SECRET, "")
    
    @property 
    def app_account_no(self) -> str:
        return os.environ.get(const.VAR_KIS_ACCOUNT_NO, "")
    
    @property
    def mock(self) -> bool:
        return os.environ.get(const.VAR_KIS_MOCK, "true").lower() == "true"


@dataclass
class LogConfig:
    base_dir: Path = field(repr=False)  # 외부 주입
    
    signal_str: str = "signal"
    signal_log_fname: str = field(init=False)
    order_str: str = "order"
    order_log_fname: str = field(init=False)

    @property
    def dir(self) -> Path:
        d = self.base_dir / const.LOG_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def __post_init__(self):
        self.signal_log_fname = f"{self.signal_str}s.jsonl"
        self.order_log_fname = f"{self.order_str}s.jsonl"


@dataclass(frozen=True)
class StoreConfig:
    base_dir: Path = field(repr=False)  # 외부 주입
    fname: str = const.STORE_FNAME
    
    load_store: str = const.LOAD_STORE
    load_kis: str = const.LOAD_KIS
    load_pykrx: str = const.LOAD_PYKRX
    
    @property 
    def dir(self) -> Path:
        d = self.base_dir / const.STORE_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d
    

@dataclass(frozen=True)
class CostConfig:
    """ 
    보수적 거래 비용 모델.

    실제 수익에 비용을 과소평가하면 백테스트가 낙관적이 되기 때문에,
    슬리피지 0.1%는 KRX 시장가 주문 기준 현실적 상한선임.

    왕복 비용(roundtrip_cost ≒ 0.41%) → 진입 신호의 최소 기대수준
    combined_score >= 0.55 임계값도 이 비용을 커버할 수 있는 신뢰도에서 도출됨
    """
    # 0.015% 편도 (증권사별로 다름)
    commission_rate: float = const.COMMISSION_RATE    
    # 매도 시 증권거래세 
    tax_rate: float = const.TAX_RATE            
     # 슬리피지 0.1% (보수적 추정)
    slippage_rate: float = const.SLIPPAGE_RATE     

    @property 
    def roundtrip_cost(self) -> float:
        """ 왕복 거래에 소요되는 총 비용 계산을 위한 수수료율 """
        return self.commission_rate * 2 + self.tax_rate + self.slippage_rate * 2


@dataclass 
class TripleBarrierConfig:
    """ 
    Triple Barrier 라벨링 임계값 설정

    백테스트 기준과 실거래 기준이 동일해야 함.
    """
    take_profit: float = const.TAKE_PROFIT
    stop_loss: float = const.STOP_LOSS
    time_horizon: int = const.TIME_HORIZON


@dataclass 
class _Config:
    # root_dir: ~/projects/mps
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    sys: SysConfig = field(default_factory=SysConfig)
    run: RunBacktestConfig = field(default_factory=RunBacktestConfig)
    log: LogConfig = field(init=False)
    cost: CostConfig = field(default_factory=CostConfig)
    kis: KisApiConfig = field(default_factory=KisApiConfig)
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)

    def __post_init__(self):
        self._data_dir = self.root_dir / const.DATA_DIR
        self.log = LogConfig(base_dir=self._data_dir)
        self.store = StoreConfig(base_dir=self._data_dir)
        

from . import _messages as messages 
# ── 전역 싱글톤 ─────────────────────────
config = _Config()

