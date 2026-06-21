from __future__ import annotations 

import os 
import math 
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field, asdict 

from ._key_strings import _KeyValues, _StringValue
from mps.core.types import AggregationMode, SignalDirection, PatternName


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
    
    # 백테스트 관련 윈도우 정보
    train_days              : int           = 30        # walk_forward 폴드별 학습 윈도우 일자
    test_days               : int           = 10        # walk_forward 테스트 윈도우 크기(2주)


# ─────────────────────────────────────
#   시스템 + 경로 정보 
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
    datetime_format         : str           = "%Y%m%d_%H%M%S"

    signal_logging_on       : bool          = True
    order_logging_on        : bool          = True


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
    
    @property 
    def monitoring(self) -> Path:
        # 모니터링 로그
        # output logger(signal, order) 디렉토리(logs)와 구분해 저장
        return self._artifacts_dir / "monitoring"

    @property 
    def store(self) -> Path:
        return self._artifacts_dir / "store"
    
    @property 
    def models(self) -> Path:
        return self._artifacts_dir / "models"
    
    @property
    def lstm_model_fpath(self) -> Path: 
        return self.models / "lstm_numeric.pt"
    
    @property 
    def cnn_model_fpath(self) -> Path:
        return self.models / "cnn_numeric.pt"
    
    def ensure_dirs(self) -> None:
        """ 산출물 관련 디렉토리 일괄 생성 ─ 설정파일 생성 시 1회(명시적으로) 실행 """
        for d in (self.monitoring, self.store, self.models):
            d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────
#   전역 싱글톤 설정
# ─────────────────────────────────────

@dataclass(frozen=True)
class _Config:

    run                     : _RunConfig    = field(default_factory=_RunConfig)
    sys                     : _SystemConfig = field(default_factory=_SystemConfig)
    path                    : _PathConfig   = field(default_factory=_PathConfig)

    key                     : _KeyValues    = field(default_factory=_KeyValues)
    str                     : _StringValue  = field(default_factory=_StringValue)


config = _Config()
config.path.ensure_dirs()   # 시스템에서 필요한 디렉토리 명시적 초기화(생성)    