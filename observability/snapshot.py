""" 
ChartSnapshotSaver — 패턴 신호 발생 시점 차트 이미지 저장
"""
from __future__ import annotations 

from datetime import datetime 
from pathlib import Path 

from mps.sys.config import settings 
from mps.data.types import Bar 


class ChartSnapshotSaver:
    def __init__(self, snapshot_dir: Path | None = None) -> None:
        self._dir = snapshot_dir or settings.snapshots_dir

    def save(
        self, ticker: str, timestamp: datetime, bars: list[Bar], pattern_name: str = ""
    ) -> Path: 
        try:
            import mplfinance as mpf
            import pandas as pd 
        except ImportError:
            return Path()
        
        df = pd.DataFrame({
            "Open": [bar.open for bar in bars],
            "High": [bar.high for bar in bars],
            "Low": [bar.low for bar in bars],
            "Close": [bar.close for bar in bars],
            "Volume": [bar.volume for bar in bars],
        }, index=pd.DatetimeIndex([bar.timestamp for bar in bars]))

        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        fname = f"{ticker}_{ts_str}_{pattern_name}.png"
        out_fpath = self._dir / fname 

        mpf.plot(
            df, 
            type="candle",
            volume=True,
            title=f"{ticker} {ts_str} [{pattern_name}]",
            savefig=str(out_fpath),
            style="charles"
        )

        return out_fpath