from __future__ import annotations 

import os 
import math 
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo 
from dataclasses import dataclass, field, asdict 

from ._key_strings import _KeyValues, _StringValue
from mps.core.types import (
    SignalDirection, PatternSource, ExitHoldReason, OrderType, OrderAction, OrderStatus
)


# ─────────────────────────────────────
#   프로그램 실행 정보
# ─────────────────────────────────────
@dataclass(frozen=True)
class _RunConfig:
    """ 프로젝트 수행 관련 설정값 정의 """
    
    # 훈련 종목
    tickers                 : tuple[str, ...] = ("005930",)

    # 시작·종료 일자
    start_date_str          : str           = "20250101"
    end_date_str            : str           = "20251231"

    # 초기 자본금
    init_capital            : float         = 10_000_000.0


# ─────────────────────────────────────
#   시스템 정보
# ─────────────────────────────────────
@dataclass(frozen=True)
class _SystemConfig:
    name                    : str           = "MPS"
    phase                   : int           = 2
    version                 : str           = "v3"
    
    seed                    : int           = 42
    
    zero                    : float         = 1e-8
    
    timezone                : ZoneInfo      = field(default_factory=lambda: ZoneInfo("Asia/Seoul"))
    date_format             : str           = "%Y%m%d"
    time_format             : str           = "%H:%M:%S"
    
    # 데이터 강제 재수집
    force_data_refresh      : bool          = False
    
    # 학습용 디바이스
    torch_device            : str           = "cuda"


# ─────────────────────────────────────
#   경로 ─ 런타임 산줄물은 소스코드와 분리해 artifacts 아래에 저장
# ─────────────────────────────────────
@dataclass(frozen=True)
class _PathConfig:
    # version root: ~/projects/mps/src/v3
    # ─ 이 아래에 mps(소스코드 루트), artifacts(산출물 루트)로 분리해 관리
    _root_dir               : Path          = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    
    # 저장소·로그 파일명
    store_fname             : str           = "minute_bars.parquet"
    signal_log_title        : str           = "signal"
    signal_log_fname        : str           = f"{signal_log_title}s.jsonl"
    order_log_title         : str           = "order"
    order_log_fname         : str           = f"{order_log_title}s.jsonl"

    @property 
    def _artifacts_dir(self) -> Path:
        return self._root_dir / "artifacts"
    
    # @property 
    # def monitoring(self) -> Path:
    #     # 모니터링 로그
    #     return self._artifacts_dir / "monitoring"

    @property 
    def output(self) -> Path:
        # output logger(signal, order) 디렉토리(logs)와 구분해 저장
        return self._artifacts_dir / "output"       
    
    @property 
    def store(self) -> Path:
        return self._artifacts_dir / "store"
    
    @property 
    def models(self) -> Path:
        return self._artifacts_dir / "models"
    
    def ensure_dirs(self) -> None:
        """ 산출물 관련 디렉토리 일괄 생성 ─ 설정파일 생성 시 1회(명시적으로) 실행 """
        for d in (self.output, self.store, self.models):
            d.mkdir(parents=True, exist_ok=True)


    # _base_dir               : Path          = field(repr=False)

    # store                   : Path          = field(init=False)
    # store_fname             : str           = "minute_bars.parquet"
    
    # logs                    : Path          = field(init=False)
    # signal_log_fname        : str           = "signals.jsonl"
    # order_log_fname         : str           = "orders.jsonl"

    # def __post_init__(self) -> None:
    #     self.store = self._base_dir / "store"
    #     self.logs = self._base_dir / "logs"
        
    #     # 현재 객체의 field중에 Path 속성을 가진 필드를 찾아서
    #     # 해당 디렉토리가 없으면 해당 디렉토리를 생성함.
    #     for f in fields(self):
    #         attr = getattr(self, f.name)
    #         if isinstance(attr, Path):
    #             attr.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────
#   데이터 상수 ─ 룩백·합성 데이터 등 데이터 관련 상수
# ─────────────────────────────────────
@dataclass(frozen=True)
class _DataConfig:
    # 캐시 데이터를 이용하지 않고 강제로 다시 데이터를 읽어오는 옵션
    force_refresh_data      : bool          = False 

    # 룩백: 신호 생성 전 반드시 확보할 과거 봉 수 (120~240분(봉))
    lookback_minutes        : int           = 120 
    # 지표 워밍업 봉 수 ─ 시뮬레이터 버퍼 = lookback + warmup_bars
    #   - MACD(EMA26+Signal9)=35봉, BB·ret_20=20봉 → 50봉이면 EMA 수렴에 충분
    #   - 버퍼 앞부분 NaN → 0 구간이 룩백 윈도우에 섞이는 것을 방지하는 역할
    warmup_bars             : int           = 50

    # pykrx 일봉 생성 시 옵션
    volume_weight_09        : float         = 2.6   # 개장 30분 거래량 비율
    volume_weight_12        : float         = 1.5   # 점심시간 거래량 비율
    volume_weight_15        : float         = 2.3   # 폐장시 거래량 비율
    min_max_noise           : float         = 0.5   # 고·저가 노이즈 비율

# ─────────────────────────────────────
#   시장 상수 ─ 한국 시장 기본 정보
# ─────────────────────────────────────
@dataclass(frozen=True)
class _MarketConfig:
    open_time_str           : str           = "09:00"       
    close_time_str          : str           = "15:30"       
    days_per_year           : int           = 242           # 연 영업일 개략값 (25년 242일) 
    minutes_per_day         : int           = 60 * 6 + 30   # = 390
    force_close_minutes     : int           = 15            # 마감 N분 전 강제 청산 (15:15)

    @property 
    def open_time(self) -> time:
        return time(int(self.open_time_str[:2]), int(self.open_time_str[3:]))
    
    @property 
    def close_time(self) -> time:
        return time(int(self.close_time_str[:2]), int(self.close_time_str[3:]))
    
    @property 
    def bars_per_day(self) -> int:
        # 일분봉은 일 분 수와 동일 ─ 5분봉 등으로 전환 시 이 부분만 수정하면 됨
        return self.minutes_per_day                         

# ── KIS API 연계 정보 ────────────────

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


# ── 모델 훈련 설정 ──────────────────

@dataclass(frozen=True)
class _LSTMConfig:
    input_size              : int           = 14
    hidden_size             : int           = 64
    num_layers              : int           = 2
    num_classes             : int           = 2     # BUY·HOLD 사냐마냐
    dropout                 : float         = 0.2
    
    def to_dict(self) -> dict:
        return asdict(self)
    

@dataclass(frozen=True)
class _CNNConfig:
    in_channels             : int           = 5
    num_classes             : int           = 2
    dropout                 : float         = 0.2
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    
@dataclass(frozen=True)
class _HyperparameterConfig:
    epochs                  : int           = 50
    batch_size              : int           = 64
    lr                      : float         = 1e-4
    weight_decay            : float         = 1e-4
    val_ratio               : float         = 0.2       # 뒤 20%를 가지고 검증
    patience                : int           = 10        # 조기 종료를 위한 검사 횟 수    
    

@dataclass
class _TrainConfig:
    lstm_settings           : _LSTMConfig   = field(default_factory=_LSTMConfig)
    cnn_settings            : _CNNConfig    = field(default_factory=_CNNConfig)

    # 하이퍼파라메터
    hyper_params            : _HyperparameterConfig = field(default_factory=_HyperparameterConfig)


# ── 전역 싱글톤 설정 ──────────────────

@dataclass 
class _Config:

    run                     : _RunConfig    = field(default_factory=_RunConfig)
    sys                     : _SystemConfig = field(default_factory=_SystemConfig)
    path                    : _PathConfig   = field(default_factory=_PathConfig)
    data                    : _DataConfig   = field(default_factory=_DataConfig)
    market                  : _MarketConfig = field(default_factory=_MarketConfig)
    kis                     : _KisApiConfig = field(default_factory=_KisApiConfig)
    train                   : _TrainConfig  = field(default_factory=_TrainConfig)
    
    key                     : _KeyValues    = field(default_factory=_KeyValues)
    str                     : _StringValue  = field(default_factory=_StringValue)


config = _Config()
config.path.ensure_dirs()   # 시스템에서 필요한 디렉토리 명시적 초기화(생성)