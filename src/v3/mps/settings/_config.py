from __future__ import annotations 

import os 
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo 
from dataclasses import dataclass, field, fields, asdict 

from ._key_strings import _KeyValues, _StringValue


# ── 프로그램 실행 정보 ─────────────────

@dataclass(frozen=True)
class _RunConfig:
    """ 프로젝트 수행 관련 설정값 정의 """
    
    # 훈련 종목
    tickers                 : list[str]     = field(default_factory=lambda: ["005930"])

    # 시작·종료 일자
    start_date_str          : str           = "20250101"
    end_date_str            : str           = "20251231"

    # 초기 자본금
    init_capital            : float         = 10_000_000.0


# ── 시스템 정보 ─────────────────────

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


@dataclass
class _PathConfig:
    _base_dir               : Path          = field(repr=False)

    store                   : Path          = field(init=False)
    store_fname             : str           = "minute_bars.parquet"
    
    logs                    : Path          = field(init=False)
    signal_log_fname        : str           = "signals.jsonl"
    order_log_fname         : str           = "orders.jsonl"

    def __post_init__(self) -> None:
        self.store = self._base_dir / "store"
        self.logs = self._base_dir / "logs"
        
        # 현재 객체의 field중에 Path 속성을 가진 필드를 찾아서
        # 해당 디렉토리가 없으면 해당 디렉토리를 생성함.
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Path):
                attr.mkdir(parents=True, exist_ok=True)


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
    hyper                   : _HyperparameterConfig = field(default_factory=_HyperparameterConfig)


# ── 전역 싱글톤 설정 ──────────────────

@dataclass 
class _Config:

    # root_dir: ~/projects/mps ─ ~/projects/mps/src/v3/mps/core/config/_config.py에서 5번째 위 부모
    _root_dir               : Path          = field(default_factory=lambda: Path(__file__).resolve().parents[5])

    run                     : _RunConfig    = field(default_factory=_RunConfig)
    sys                     : _SystemConfig = field(default_factory=_SystemConfig)
    path                    : _PathConfig   = field(init=False)
    kis                     : _KisApiConfig = field(default_factory=_KisApiConfig)
    train                   : _TrainConfig  = field(default_factory=_TrainConfig)
    
    key                     : _KeyValues    = field(default_factory=_KeyValues)
    str                     : _StringValue  = field(default_factory=_StringValue)

    def __post_init__(self) -> None:
        # 런타임 산출물(로그, 입력 데이터, 모델 가중치 데이터) 저장 기본 폴더
        self._base_dir                      = self._root_dir / "artifacts" / self.sys.version
        self.path                           = _PathConfig(_base_dir=self._base_dir)


config = _Config()