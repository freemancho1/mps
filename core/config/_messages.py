from __future__ import annotations 

from dataclasses import dataclass 
from typing import Callable, TypeAlias

from mps.core.libs import CF 


mfn: TypeAlias = Callable[..., str]     # Message_FuNction, str과 자리수 맞추기 위해 3글자로 함


@dataclass(frozen=True)
class _StoreMessages:
    file_not_found_err      : mfn = lambda path: CF(f"Parquet 파일이 존재하지 않습니다: {path}")
    load_data_info          : mfn = lambda df: CF(f"불러온 데이터 정보: 데이터 크기={df.shape}, 기간={df.index[0].date()}~{df.index[-1].date()}")
    save_parquet_info       : mfn = lambda df: CF(f"저장한 데이터 정보: 데이터 크기={df.shape}, 기간={df.index[0].date()}~{df.index[-1].date()}")


@dataclass(frozen=True)
class _LoaderMessages:
    load_result             : mfn = lambda soc, bars: CF(f"데이터 로드 결과: 출처[{soc}], 크기[{len(bars)}], 기간: {bars[0].timestamp} ~ {bars[-1].timestamp}")


@dataclass(frozen=True)
class _FeatureMessages:
    labeling_result         : mfn = lambda bars, label_dist: CF(f"라벨링 대상 분봉 갯 수: {len(bars)}개, 라벨링 결과: {label_dist}")
    ds_window_size          : mfn = lambda s_pit, e_pit: CF(f"데이터셋 학습 윈도우: 시작={s_pit}, 종료={e_pit}")


@dataclass(frozen=True)
class _Messages:
    store                   : _StoreMessages = _StoreMessages()
    loader                  : _LoaderMessages = _LoaderMessages()
    feature                 : _FeatureMessages = _FeatureMessages()


messages                    : _Messages = _Messages()