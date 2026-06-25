from __future__ import annotations

import inspect
from datetime import datetime 


class _ColorLogger:
    """ 레벨별 색상 출력 전용 로거. """
    _RESET = "\033[0m"

    _COLORS: dict[str, str] = {
        "DEBUG"   : "\033[38;5;245m",  # 중간 회색
        "INFO"    : "\033[1m",          # 밝은 흰색 (bold)
        "POINT"   : "\033[38;5;39m",   # 밝은 파랑
        "WARNING" : "\033[93m",         # 연노랑
        "ERROR"   : "\033[91m",         # 연빨강
        "CRITICAL": "\033[95m",         # 연보라
        "TEST"    : "\033[38;5;218m",   # 연핑크
    }
    _DT_FMT = "%Y-%m-%d %H:%M:%S"
    
    def __init__(self) -> None:
        self._call_func = None 

    def _print(self, level: str, message: object) -> None:
        now   = datetime.now()
        ts    = now.strftime(self._DT_FMT)
        ms    = f"{now.microsecond // 1000:03d}"
        color = self._COLORS.get(level, "")
        print(f"{color}{ts}.{ms} {level:>8}: {message}{self._RESET}")

    def debug(self, message: object) -> None:
        self._print("DEBUG", message)

    def info(self, message: object) -> None:
        self._print("INFO", message)

    def point(self, message: object) -> None:
        self._print("POINT", message)

    def warning(self, message: object) -> None:
        self._print("WARNING", message)

    def error(self, message: object) -> None:
        self._print("ERROR", message)

    def critical(self, message: object) -> None:
        self._print("CRITICAL", message)
        
    def test(self, message: object) -> None:
        frame = inspect.stack()[1]
        call_func = frame.function

        caller_self = frame.frame.f_locals.get("self")
        class_name = type(caller_self).__name__ if caller_self is not None else None
        label = f"{class_name}.{call_func}" if class_name else call_func

        if self._call_func != call_func:
            self._call_func = call_func
            print()
            self._print("TEST", label)
            
        self._print("TEST", message)


color_logger = _ColorLogger()