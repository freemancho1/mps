from __future__ import annotations 

import os 
import math 
import torch
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo 
from dataclasses import dataclass, field, asdict

from mps.core.types import Direction, BSDirection, PatternSource
from mps.core.types import OrderType, OrderStatus, ExitHoldReason


@dataclass 
class _RunConfig: 
    # 초기 입력 정보
    tickers: list[str]          = field(default_factory=lambda: ["005930"])
    start_date_str: str         = "20250101"
    end_date_str: str           = "20251231"
    init_capital: float         = 10_000_000.0
    test_days: int              = 10                    # walk_forward의 테스트 윈도우 일자(2주)
    
    # Model Training Config
    torch_device: str           = "cuda" if torch.cuda.is_available() else "cpu"
    numeric_track: str          = "numeric"
    pattern_track: str          = "pattern"
                                                        # 방향성 없음
                                                        # tuple 자체는 field로 감싸지 않아도 되지만,
                                                        # tuple 속에 있는 {}가 나중에 공유될 수 있어 감쌈.
    no_signal: tuple[Direction, float, dict] = field(default_factory=lambda: ("HOLD", 0.0, {}))    
    no_signal_pattern: tuple[Direction, float, str] = field(default_factory=lambda: ("HOLD", .0, "none"))
                                                        # 14개 생성 피처 ─ 순서 변경되면 안 됨
                                                        # 실제 정의는 _Config에서 정의됨
    feature_names: list[str]    = field(default_factory=list) 
    feature_idx: dict[str, int] = field(init=False)
    
    # 데이터 읽어오는 방식
    force_data_refresh: bool    = False                 # 강제로 데이터 읽어오기: 읽어오지 않음(Fasle)

    # 일자 정보
    days_per_year: int          = 242                   # 임의로 잡은 날짜이고, krx에서 실 거래일자만 가져옴(25년도 242일)
    minutes_per_day: int        = 60 * 6 + 30           # 09:00 ~ 15:30 = 390분
    bars_per_day: int           = field(init=False)     # (분봉 기준이니) minutes_per_day와 같은 값 
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

    pc: _PatternConfidence      = field(init=False)

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
    stop_loss: float            = 0.01                  # -1.0%
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
        self.bars_per_day = self.minutes_per_day


@dataclass 
class _PatternConfidence:
    single_peak: float          = 0.6       # 단봉
    double_peak: float          = 0.7       # 이중봉
    triple_peak: float          = 0.75      # 삼봉
    chart: float                = 0.65      # 차트


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
class _ModelConfig:
    base_dir: Path              = field(repr=False)
    lstm_model_fpath: Path      = field(init=False)
    cnn_model_fpath: Path       = field(init=False)
    
    _model_dir: str             = "models"
    _lstm_fname: str            = "lstm_numeric.pt"
    _cnn_fname: str             = "cnn_pattern.pt"
    
    def __post_init__(self) -> None:
        self.lstm_model_fpath = self.base_dir / self._model_dir / self._lstm_fname
        self.cnn_model_fpath = self.base_dir / self._model_dir / self._cnn_fname
        

@dataclass 
class _TrainConfig:
    epochs: int                 = 40
    batch_size: int             = 64
    lr: float                   = 1e-4
    weight_decay: float         = 1e-4
    val_ratio: float            = 0.2       # 뒤 20%를 검증에 사용
    patience: int               = 10        # 조기 종료 인내심
    seed: int                   = field(init=False)
    device: str                 = field(init=False)
    

@dataclass 
class _KeyConfig:               # 알파벳 순
    atr_14: str                 = "atr_14"
    
    bb_lower: str               = "bb_lower"
    bb_mid: str                 = "bb_mid"
    bb_pband: str               = "bb_pband"
    bb_upper: str               = "bb_upper"
    BUY: Direction | BSDirection = "BUY"
    
    capital: str                = "--capital"
    close: str                  = "close"
    count: str                  = "count"
    
    end: str                    = "--end"

    feature: str                = "feature"
    
    high: str                   = "high"
    HOLD: Direction | BSDirection | ExitHoldReason = "HOLD"
    
    low: str                    = "low"
    
    macd: str                   = "macd"
    macd_diff: str              = "macd_diff"
    macd_signal: str            = "macd_signal"
    max_ms: str                 = "max_ms"
    mean_ms: str                = "mean_ms"

    numeric: str                = "numeric"
    
    obv: str                    = "obv"
    open: str                   = "open"
    
    pattern: str                = "pattern"
    p95_ms: str                 = "p95_ms"
    
    ret_1: str                  = "ret_1"
    ret_5: str                  = "ret_5"
    ret_20: str                 = "ret_20"
    rsi_14: str                 = "rsi_14"
    
    SELL: Direction | BSDirection = "SELL"
    start: str                  = "--start"
    state_dict: str             = "state_dict"
    
    test_days: str              = "--test_days"
    ticker: str                 = "--ticker"
    
    volume: str                 = "volume"
    volume_ratio: str           = "volume_ratio"
    

@dataclass 
class _StrConfig: 
    bearish_engulfing: str      = "BEARISH_ENGULFING"
    box_breakout_down: str      = "BOX_BREAKOUT_DOWN"
    box_breakout_up: str        = "BOX_BREAKOUT_UP"
    bullish_engulfing: str      = "BULLISH_ENGULFING"
    cancelled: OrderStatus      = "CANCELLED"
    evening_star: str           = "EVENING_STAR"
    filled: OrderStatus         = "FILLED"
    force_close: ExitHoldReason = "FORCE_CLOSE"
    hammer: str                 = "HAMMER"
    hold: ExitHoldReason        = "HOLD"
    market: OrderType           = "MARKET"
    morning_star: str           = "MORNING_STAR"
    rule: PatternSource         = "RULE"
    shooting_star: str          = "SHOOTING_STAR"
    stop_loss: ExitHoldReason   = "STOP_LOSS"
    take_profit: ExitHoldReason = "TAKE_PROFIT"
    time_out: ExitHoldReason     = "TIME_OUT"
    

@dataclass 
class _LSTMConfig:
    input_size: int             = 14
    hidden_size: int            = 64
    num_layers: int             = 2
    num_classes: int            = 3         # BUY·SELL·HOLD
    dropout: float              = 0.2
    
    def to_dict(self) -> dict:
        return asdict(self)
    

@dataclass 
class _CNNConfig:
    in_channels: int            = 5
    num_classes: int            = 3
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
    model: _ModelConfig = field(init=False)
    key: _KeyConfig = field(default_factory=_KeyConfig)
    str: _StrConfig = field(default_factory=_StrConfig)
    lstm: _LSTMConfig = field(default_factory=_LSTMConfig)
    cnn: _CNNConfig = field(default_factory=_CNNConfig)
    train: _TrainConfig = field(default_factory=_TrainConfig)

    def __post_init__(self) -> None:
        self._data_dir = self._root_dir / "data"
        self.log = _LogConfig(base_dir=self._data_dir)
        self.store = _StoreConfig(base_dir=self._data_dir)
        self.model = _ModelConfig(base_dir=self._data_dir)
        
        self.run.feature_names = [
            self.key.rsi_14, self.key.macd, self.key.macd_signal, self.key.macd_diff,
            self.key.bb_upper, self.key.bb_mid, self.key.bb_lower, self.key.bb_pband,
            self.key.obv, self.key.atr_14, self.key.volume_ratio,
            self.key.ret_1, self.key.ret_5, self.key.ret_20,
        ]
        self.run.feature_idx = {name: idx for idx, name in enumerate(self.run.feature_names)}
        self.run.pc = _PatternConfidence()
        
        self.train.seed = self.run.seed
        self.train.device = self.run.torch_device


config = _Config()