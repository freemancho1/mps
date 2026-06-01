from __future__ import annotations 

import os 
import math 
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo 
from dataclasses import dataclass, field, asdict

from mps.core.types import Direction 


@dataclass 
class _RunConfig: 
    # 초기 입력 정보
    tickers: list[str]          = field(default_factory=lambda: ["005930"])
    start_date_str: str         = "20250101"
    end_date_str: str           = "20251231"
    init_capital: float         = 10_000_000.0
    
    # Model Training Config
    torch_device: str           = "gpu"

    no_signal: tuple[Direction, float, dict] = ("HOLD", 0.0, {})

    # 데이터 읽어오는 방식
    force_data_refresh: bool    = False                 # 강제로 데이터 읽어오기: 읽어오지 않음(Fasle)

    # 일자 정보
    days_per_year: int          = 252                   # 임의로 잡은 날짜이고, krx에서 실 거래일자만 가져옴(25년도 242일)
    minutes_per_day: int        = 60 * 6 + 30           # 09:00 ~ 15:30 = 390분
    force_close_minutes: int    = 15                    # 강제 종료 시간

    # 시간 정보
    open_time_str: str          = "09:00"
    open_time: time             = field(init=False)
    close_time_str: str         = "15:30"
    close_time: time            = field(init=False)
    # 시간대별 거래량 비중
    volume_weight_09: float     = 2.6                   # 2.6 배
    volume_weight_12: float     = 1.5
    volume_weight_15: float     = 2.3
    # 고가/저가: 시가·종가 범위에 작은 노이즈 추가 (현실적 캔들 형태)
    min_max_noise: float        = 0.5

    # 룩백 윈도우: 신호 생성전 반드시 확보해야 할 과거 봉 수
    lookback_minutes: int       = 120
    buffer_days: int            = field(init=False)     # ceil(120 / 390) + 1 = 2

    # RSI 정보
    rsi_oversold: float         = 35.0
    rsi_overbought: float       = 65.0
    rsi_closeover_base: float   = 0.3

    # 결합 정보
    numeric_cw: float           = 0.5                   # numeric combined weight
    pattern_cw: float           = 0.5                   # pattern combined weight

    # 비용 정보
    commission_rate: float      = 0.00015               # 거래 비용(증권사별로 다름: 0.015%)
    tax_rate: float             = 0.0018                # 매도세(0.18%)
    slippage_rate: float        = 0.001                 # 슬리피지 비율(0.1%)

    @property
    def roundtrip_cost_rate(self) -> float:
        """ 왕복 거래에 소요되는 총 수수료율 = 0.41% """
        return (self.commission_rate + self.slippage_rate) * 2 + self.tax_rate

    # Triple Barrier 라벨링 임계값 설정
    take_profit: float          = 0.02                  # 2.0%
    stop_loss: float            = 0.005                 # -0.5%
    time_horizon: int           = 60                    # 60분

    # 거래 제약조건
    max_latency_ms: float       = 5000.0                # 최대 지연 시간 (5초)
    min_combined_score: float   = 0.55                  # 거래를 위한 최소 결합 점수 
    max_capital_pct: float      = 0.1                   # 거래 별 거래 금액 비율(= 초기 자본의 10%)

    # 시스템 정보
    ## 기본 정보
    seed: int                   = 42
    phase: int                  = 2
    ## 숫자 관련 
    zero: float                 = 1e-8
    ## 시간 관련
    timezone: ZoneInfo          = field(default_factory=lambda: ZoneInfo("Asia/Seoul"))
    date_format: str            = "%Y%m%d"

    def __post_init__(self) -> None:
        self.buffer_days = math.ceil(self.lookback_minutes / self.minutes_per_day) + 1
        self.open_time = time(int(self.open_time_str[:2]), int(self.open_time_str[3:]))
        self.close_time = time(int(self.close_time_str[:2]), int(self.close_time_str[3:]))


@dataclass 
class _KisApiConfig:
    
    @property 
    def app_key(self) -> str:
        return os.environ.get("KIS_APP_KEY", "")
    
    @property 
    def app_secret(self) -> str: 
        return os.environ.get("KIS_APP_SECRET", "")
    
    @property 
    def app_account_no(self) -> str:
        return os.environ.get("KIS_ACCOUNT_NO", "")
    
    @property
    def mock(self) -> bool:
        return os.environ.get("KIS_MOCK", "true").lower() == "true"
    

@dataclass 
class _LogConfig:
    base_dir: Path              = field(repr=False)                  # 외부 주입
    dir: Path                   = field(init=False)

    title_signal: str           = "signal"
    signal_log_fname: str       = field(init=False)
    title_order: str            = "order"
    order_log_fname: str        = field(init=False)
    file_ext: str               = ".jsonl"

    def __post_init__(self) -> None:
        self.dir = self.base_dir / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.signal_log_fname = f"{self.title_signal}s{self.file_ext}"
        self.order_log_fname = f"{self.title_order}s{self.file_ext}"


@dataclass 
class _StoreConfig:
    base_dir: Path              = field(repr=False)
    dir: Path                   = field(init=False)
    fname: str                  = "minute_bars.parquet"

    load_store: str             = "STORE"
    load_kis: str               = "KIS"
    load_pykrx: str             = "PYKRX"

    def __post_init__(self) -> None:
        self.dir = self.base_dir / "store"
        self.dir.mkdir(parents=True, exist_ok=True)


@dataclass 
class _KeyConfig:
    ticker: str                 = "--ticker"
    start: str                  = "--start"
    end: str                    = "--end"
    capital: str                = "--capital"
    test_days: str              = "--test_days"


@dataclass 
class _LSTMConfig:
    input_size: int             = 14
    hidden_size: int            = 64
    num_layers: int             = 2
    num_classes: int            = 3         # BUY·SELL·HOLD
    dropout: float              = 0.2
    
    def to_dict(self) -> dict:
        return asdict(self)


# ── 전역 싱글톤 ─────────────────────────
@dataclass 
class _Config:
    # root_dir: ~/projects/mps
    _root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    run: _RunConfig = field(default_factory=_RunConfig) 
    kis: _KisApiConfig = field(default_factory=_KisApiConfig)
    log: _LogConfig = field(init=False)
    store: _StoreConfig = field(init=False)
    key: _KeyConfig = field(default_factory=_KeyConfig)
    lstm: _LSTMConfig = field(default_factory=_LSTMConfig)

    def __post_init__(self) -> None:
        self._data_dir = self._root_dir / "data"
        self.log = _LogConfig(base_dir=self._data_dir)
        self.store = _StoreConfig(base_dir=self._data_dir)


config = _Config()