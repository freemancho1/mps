#!/home/freeman/anaconda3/envs/aidds_develop/bin/python

import os 
import re 
import time 
import argparse 

from colorama import Fore, Style, init 
from typing import List, Dict, Pattern 

# colorama 초기화: Windows 환경에서 색상 출력을 위해 필요
init(autoreset=True)

# ----------------------
# 상수 및 색상 정의
# ----------------------

# 로그 레벨 분석 정규 표현식 정의
LOG_LEVEL_PATTERN: Pattern = re.compile(r'\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]')

# 로그 레벨별 색상 정의
COLOR_MAP: Dict[str, str] = {
    # 'DEBUG'     : Fore.BLACK + Style.BRIGHT,
    'DEBUG'     : Fore.WHITE,
    'INFO'      : Fore.RESET + Style.BRIGHT, # 밝은 흰색
    'WARNING'   : Fore.LIGHTYELLOW_EX, 
    'ERROR'     : Fore.LIGHTRED_EX, 
    'CRITICAL'  : Fore.LIGHTMAGENTA_EX,
}
# 매칭되지 않는 색상 지정
DEFAULT_COLOR = Fore.WHITE


# ----------------------
# 함수 정의
# ----------------------

def colorize_log_line(line: str) -> str:
    """ 로그 레벨에 따라 줄의 색상을 변경 """
    color = DEFAULT_COLOR
    if match := LOG_LEVEL_PATTERN.search(line):
        level = match.group(1)
        color = COLOR_MAP.get(level, DEFAULT_COLOR)
        
    return f"{color}{line}{Style.RESET_ALL}"

def read_last_n_lines(filepath: str, line_count: int) -> List[str]:
    """ 파일의 끝에서부터 line_count줄을 메모리 효율적으로 읽어 옴 """
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return []
    
    # 한번에 읽을 블럭 크기 정의
    block_size = 4096
    lines: List[str] = []
    
    # 파일을 "rb" 즉 바이너리로 읽어 인코딩 문제를 피하고 seek의 정확성을 높임
    with open(filepath, "rb") as file:
        # 파일 끝에서부터 역순으로 블록을 탐색
        seek_pos = file_size 
        while seek_pos > 0 and len(lines) < line_count:
            # 읽을 블록 크기 결정
            read_size = min(block_size, seek_pos)
            seek_pos -= read_size 
            file.seek(seek_pos)
            
            # 블록 읽기
            buffer = file.read(read_size)
            
            # 읽은 버퍼를 현재 시스템의 기본 인코딩(혹은 utf-8)으로 디코딩
            content = buffer.decode("utf-8", errors="ignore")
            # 줄 단위로 분리
            new_lines = content.splitlines()
            # 기존 줄 리스트 앞에 추가
            lines = new_lines + lines 
        
        # 최종적으로 N개 이상의 줄이 수집되었을 경우, 마지막 line_count개만 반환
        return lines[-line_count:]
        

def follow_file(filepath: str, initial_lines: int):
    """ 실시간 파일 모니터링 함수 (tail -f 기능) """
    
    if not os.path.exists(filepath):
        print(f"{Fore.RED}Error: 파일을 찾을 수 없습니다({filepath})")
        return 
    
    try:
        # 파일에서 기본 라인수의 리스트를 효율적으로 읽어와 출력
        initial_contents = read_last_n_lines(filepath, line_count=initial_lines)
        for line in initial_contents:
            print(colorize_log_line(line.strip()))
            
        # 파일의 끝으로 이동하여 모니터링 시작
        with open(filepath, "r", encoding="utf-8") as file:
            file.seek(0, os.SEEK_END)
        
            while True:
                new_line = file.readline()
                if new_line:
                    print(colorize_log_line(new_line.strip()))
                else:
                    time.sleep(0.3)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}:: 모니터링 종료")
    except Exception as e:
        print(f"\n{Fore.RED}예상치 못한 오류 발생: {e}")


# ----------------------
# main() 함수
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="실시간 로그 파일 모니터링 및 로그 레벨 색생 출력"
    )
    # 파일 경로 지정
    parser.add_argument(
        "filepath", type=str, nargs="?", help="모니터링 할 로그파일 경로"
    )
    # 초기 출력 라인수 지정
    parser.add_argument(
        "-n", "--lines", type=int, default=10, help="초기에 출력할 파일의 마지막 줄 수(기본값 10)"
    )
    
    args = parser.parse_args()
    follow_file(args.filepath, args.lines)
    