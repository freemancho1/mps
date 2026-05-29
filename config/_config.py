from __future__ import annotations 

import os 
import math 
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo 
from dataclasses import dataclass, field 


@dataclass 
class _RunConfig: 
    tickers: list[str]      = field(default_factory=lambda: ["005930"])
    start_date: str         = "20250101"
    end_date: str           = "20251231"
    init_capital: float     = 10_000_000.0
    


# ── 전역 싱글톤 ─────────────────────────
@dataclass 
class _Config:
    # root_dir: ~/projects/mps
    _root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    run: _RunConfig = field(default_factory=_RunConfig) 

config = _Config()