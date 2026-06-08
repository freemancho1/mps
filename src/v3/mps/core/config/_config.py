from __future__ import annotations 

from genericpath import exists
import os 
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo 
from dataclasses import dataclass, field, asdict 


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


@dataclass(frozen=True)
class _SystemConfig:
    phase                   : int           = 2
    version                 : str           = "v3"


@dataclass
class _PathConfig:
    _base_dir               : Path          = field(repr=False)

    store                   : Path          = field(init=False)
    store_fname             : str           = "minute_bars.parquet"

    def __post_init__(self) -> None:
        self.store = self._base_dir / "store"
        self.store.mkdir(parents=True, exist_ok=True)



# ── 전역 싱글톤 설정 ──────────────────
@dataclass 
class _Config:

    # root_dir: ~/projects/mps ─ ~/projects/mps/src/v3/mps/core/config/_config.py에서 5번째 위 부모
    _root_dir               : Path          = field(default_factory=lambda: Path(__file__).resolve().parents[5])

    run                     : _RunConfig    = field(default_factory=_RunConfig)
    sys                     : _SystemConfig = field(default_factory=_SystemConfig)
    path                    : _PathConfig   = field(init=False)

    def __post_init__(self) -> None:
        # 런타임 산출물(로그, 입력 데이터, 모델 가중치 데이터) 저장 기본 폴더
        self._base_dir                      = self._root_dir / "artifacts" / self.sys.version
        self.path                           = _PathConfig(_base_dir=self._base_dir)


config = _Config()