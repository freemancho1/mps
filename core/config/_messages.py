from __future__ import annotations 

from dataclasses import dataclass 
from typing import Callable, TypeAlias

from mps.core.utils import CF 


mfn: TypeAlias = Callable[..., str]     # Message_FuNction, str과 자리수 맞추기 위해 3글자로 함


@dataclass(frozen=True)
class _StoreMessages:
    file_not_found_err      : mfn = lambda path: CF(f"Parquet 파일이 존재하지 않습니다: {path}")
    load_data_info          : mfn = lambda df: CF(f"불러온 데이터 정보: 데이터 크기={df.shape}, 기간={df.index[0]}~{df.index[-1]}")
    save_parquet_info       : mfn = lambda df: CF(f"저장한 데이터 정보: 데이터 크기={df.shape}, 기간={df.index[0]}~{df.index[-1]}")

@dataclass(frozen=True)
class _Messages:
    store                   : _StoreMessages = _StoreMessages()


messages                    : _Messages = _Messages()