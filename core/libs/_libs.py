import inspect
import numpy as np
import torch
from datetime import datetime 


def serialized(object):
    """ datetime 형태의 데이터를 JSON 직렬화가 가능한 dict/str 형태로 변환 """
    if isinstance(object, datetime):
        return object.isoformat()
    if hasattr(object, "__dict__"):
        return {key: serialized(value) for key, value in vars(object).items()}
    return object


def call_function(message: str) -> str:
    """ 
    해당 람다함수를 호출한 함수 이름과 라인번호를 리턴함 
    
    - lambda형 로그 메시지 출력 시 사용하며, 
      해당 람다 함수를 호출한 함수명과 라인을 리턴함
    """
    frame = inspect.stack()[2]
    location = f"{frame.filename}[{frame.lineno}]:"
    
    SIZE = 25
    location = location[-SIZE:] if len(location) > SIZE else location.rjust(SIZE)
    return f"{location} {message}"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)