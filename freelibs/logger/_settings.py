from __future__ import annotations 

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Pattern
from colorama import Fore, Style

from mps.freelibs import DictDot


log_messages = DictDot(
    file_not_found      = lambda fpath: f"\n{Fore.RED}ERROR: 로그 파일을 찾을 수 없습니다 ─ {fpath}",
    exit                = f"\n{Fore.YELLOW}로그 모니터링을 종료합니다.",
    error               = lambda err: f"\n{Fore.RED}ERROR: 예상치 못한 오류가 발생했습니다.\n{str(err)}",
)


@dataclass
class _Config:
    name                    : str           = "freelibs"
    
    out_device_file         : str           = "file"
    out_device_screen       : str           = "screen"
    out_device              : str           = out_device_file

    # dir : 아래 프로퍼티로 정의
    dir                     : Path          = field(default_factory=lambda: Path(__file__).parent / "logs")
    log_fname               : str           = f"{name}.log"

    debug                   : str           = "DEBUG"
    info                    : str           = "INFO"
    warning                 : str           = "WARNING"
    error                   : str           = "ERROR"
    critical                : str           = "CRITICAL"
    level                   : str           = debug
    level_pattern           : Pattern       = re.compile(r'\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]')

    string_format           : str           = "%(asctime)s.%(msecs)05d [%(levelname)s] %(message)s"
    datetime_format         : str           = "%Y-%m-%d %H:%M:%S"

    # 한번에 읽는 블럭 크기
    buffer_size             : int           = 4096

    sys_encoding            : str           = "utf-8"
    err_type                : str           = "ignore"

    # ANSI 256색상: 참조용 ─ colorama에는 회색이 없어서 이걸로 대체함
    gray_dark               : str           = "\033[38;5;240m"   # 짙은 회색
    gray_mid                : str           = "\033[38;5;245m"   # 중간 회색  ← DEBUG에 적합
    gray_light              : str           = "\033[38;5;250m"   # 밝은 회색
    color_map               : dict[str, str]= field(default_factory=lambda: {
        # 'DEBUG'           : Fore.BLACK + Style.BRIGHT,
        'DEBUG'             : "\033[38;5;245m", # gray_mid,
        # 'DEBUG'             : Fore.WHITE,
        'INFO'              : Fore.RESET + Style.BRIGHT, # 밝은 흰색
        'WARNING'           : Fore.LIGHTYELLOW_EX, 
        'ERROR'             : Fore.LIGHTRED_EX, 
        'CRITICAL'          : Fore.LIGHTMAGENTA_EX,        
    })
    default_color           : str           = Fore.WHITE


log_config = _Config()
log_config.dir.mkdir(parents=True, exist_ok=True)