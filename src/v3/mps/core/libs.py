import torch 
import numpy as np 
from datetime import datetime 


def serialized(object):
    """ datetime 형태의 데이터를 JSON 직렬화가 가능한 dict/str 형태로 변환 """
    if isinstance(object, datetime):
        return object.isoformat()
    if hasattr(object, "__dict__"):
        return {key: serialized(value) for key, value in vars(object).items()}
    return object

def set_seed(seed: int) -> None:
    """ 재현 가능성: numpy·torch 시드 고정 (torch는 지연 임포트 ─ 의존성 격리) """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_float(value) -> float:
    assert value is not None 
    return float(value)

def to_int(value) -> int:
    assert value is not None 
    return int(value)