from __future__ import annotations 

from datetime import datetime 
from typing import Any


DICT_KEY = "__dict__"


def serialized(obj: Any) -> Any:
    """ datetime 형태의 데이터를 json에 직렬화 가능한 형태로 변환 """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, DICT_KEY):
        return {key: serialized(value) for key, value in vars(obj).items()}
    return obj


def set_seed(seed: int) -> None:
    """ 재현 가능성 확보, 지연된 임포트 """
    import torch 
    import numpy as np 
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_float(value: Any) -> float:
    return float(value)

def to_int(value: Any) -> int:
    return int(value)