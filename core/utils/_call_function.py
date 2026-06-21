from __future__ import annotations 

import inspect 


MAX_SIZE: int = 25
MESSAGE_ONLY: bool = True 
POS_CALLED_FUNC: int = 2

def call_function(message: str) -> str:
    """ 이 함수를 호출한 함수의 부모 함수의 이름과 라인번호를 리턴함. """
    frame = inspect.stack()[POS_CALLED_FUNC]
    location = f"{frame.filename}[{frame.lineno}]: "

    location = location[-MAX_SIZE:] \
        if len(location) > MAX_SIZE else location.rjust(MAX_SIZE)
    location = "" if MESSAGE_ONLY else location

    return f"{location}{message}"