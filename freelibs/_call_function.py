import inspect

def call_function(message: str) -> str:
    """ 
    해당 람다함수를 호출한 함수 이름과 라인번호를 리턴함 
    
    - lambda형 로그 메시지 출력 시 사용하며, 
      해당 람다 함수를 호출한 함수명과 라인을 리턴함
    """
    frame = inspect.stack()[2]
    location = f"{frame.filename}[{frame.lineno}]:"
    
    SIZE: int = 25
    location = location[-SIZE:] if len(location) > SIZE else location.rjust(SIZE)
    
    IS_LOC: bool = False 
    location = f"{location} " if IS_LOC else ""
    return f"{location}{message}"