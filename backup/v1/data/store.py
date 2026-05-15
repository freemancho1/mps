"""
LocalParquetStore: DataStorePort의 1단계 구현체

[역할]
    - 분봉 데이터를 로컬 Parquet 파일로 저장·로드
    - pykrx로 수집한 데이터를 캐싱하여 반복 실행  시 재수집을 방지

[파일 구조]
    - mps/data/store/{ticker}/minute_bars.parquet
      → timestamp 인덱스, OHLCV 컬럼(Open, High, Low, Close, Volume)
"""
from __future__ import annotations

from datetime import datetime 
from pathlib import Path 
import pandas as pd 

from mps.data.types import Bar 
from mps.sys.config import settings


class LocalParquetStore:
    """ 종목별 분봉 데이터를 Parquet 파일로 저장·읽어옴. """

    def __init__(self, base_dir: Path | None = None) -> None:
        # base_dir 미지정 시 settings.data_dir 참조
        self._base = base_dir or settings.data_dir

    def _path(self, ticker: str) -> Path:
        parquet_path = self._base / ticker
        parquet_path.mkdir(parents=True, exist_ok=True)
        return parquet_path / "minute_bars.parquet"
    
    def save_bars(self, bars: list[Bar]) -> None:
        """ Bar 리스트를 Parquet에 저장 (기존 데이터와 병합, 중복 제거) 
        
            [중복처리 전략]
            - 동일 timestamp가 있으면 새 데이터(keep="last")로 변경.
              (pykrx 재수집시 최신 데이터로 덮어쓰는 것을 허용)
        """
        if not bars:
            return
        
        ticker = bars[0].ticker

        # Bar dataclass → DataFrame으로 변환 (vars()로 필드명 유지)
        new_df = pd.DataFrame([vars(b) for b in bars])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
        # timestamp를 인덱스로 설정 → 시계열 슬라이싱 가능하게 함
        new_df = new_df.set_index("timestamp").sort_index()

        path = self._path(ticker)
        print(f"save_bars path: {path}")
        # 기존 파케잍 파일이 있는 경우 병합 후 중복 제거
        if path.exists():
            old_df = pd.read_parquet(path)
            combined = pd.concat([old_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined.to_parquet(path)
        else:
            new_df.to_parquet(path)

    def load_bars(
        self,
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> list[Bar]:
        """ 지정 구간의 Bar 리스트를 Parquet에서 읽어 반환함. 
        
            - 파일이 없으면 빈 리스트를 반환하고, 이를 근거로 HistoricalDataLoader가 수집을 트리거.
            - timestamp 마스킹으로 불필요한 메모리 사용 절감.
            - 반환된 Bar들은 모두 is_complete=True (저장 시점에 완성된 봉만 사용).
        """
        path = self._path(ticker=ticker)
        if not path.exists():
            return []
        
        parquet_df = pd.read_parquet(path)
        parquet_df.index = pd.to_datetime(parquet_df.index)
        # 구간 필터링: start_date 이상 ~ end_date 이하
        mask = \
            (parquet_df.index >= pd.Timestamp(start_date)) & \
            (parquet_df.index <= pd.Timestamp(end_date))
        sub_df = parquet_df.loc[mask]

        # DataFrame을 list[Bar] 객체로 반환
        return [
            Bar(
                ticker=ticker,
                timestamp=row.name.to_pydatetime(), # Index → Python datetime
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                is_complete=True,                   # 캐시된 데이터는 항상 완성된 봉으로 처리
            ) for _, row in sub_df.iterrows()
        ]

    def list_tickers(self) -> list[str]:
        """ 저장된 종목 코드 목록을 반환 (data/store/하위 디렉토리 이름(ticker)) """
        return [p.name for p in self._base.iterdir() if p.is_dir()]

