#!/home/freeman/anaconda3/envs/mpsdev/bin/python

import os 
import time 
import argparse 
from colorama import init, Style
from pathlib import Path

from ._logger_settings import config as cfg
from ._logger_settings import messages as msg



# colorama 초기화 ─ Window 환경에서 색상 출력을 위해 필요
init(autoreset=True)


def colorize_log_line(line: str) -> str:
    """ 로그 레벨에 따라 줄의 색상을 변경 """
    if match := cfg.level_pattern.search(line):
        level = match.group(1)
        color = cfg.color_map.get(level, cfg.default_color)
    else:
        color = cfg.default_color

    return f"{color}{line}{Style.RESET_ALL}"

def read_last_n_lines(fpath: Path, lcount: int) -> list[str]:
    """ 파일의 끝에서 lcount 줄을 읽어 메모리를 효율적으로 사용함 """
    if not (fpath and fpath.exists()):
        print(msg.file_not_found(fpath))
        return []
    
    fsize: int = fpath.stat().st_size   # 파일의 크기 
    lines: list[str] = []
    # 파일을 "rb" 즉 바이너리로 읽어 인코딩 문제를 피하고, seek의 정확성을 높임
    with open(fpath, "rb") as file:
        # 파일 끝에서부터 역순으로 블록을 탐색
        seek_pos = fsize
        while seek_pos > 0 and len(lines) < lcount:
            # 읽을 블록 크기 결정
            read_size = min(cfg.buffer_size, seek_pos)
            seek_pos -= read_size 
            file.seek(seek_pos)

            # 블록 읽기
            buffer = file.read(read_size)
            # 버퍼 내용을 시스템 인코딩 방식으로 디코딩
            content = buffer.decode(cfg.sys_encoding, errors=cfg.err_type)
            # 줄 단위 분리
            new_lines = content.splitlines()
            # 기존 리스트 앞에 추가
            lines = new_lines + lines 

        # 최종적으로 N개 이상의 줄이 수집되었을 경우, 마지막 lcount개만 반환
        return lines[-lcount:]
    
def follow_file(fpath: Path, init_lines: int) -> None:
    """ 실시간 모니터링 함수 """
    if not (fpath and fpath.exists()):
        print(msg.file_not_found(fpath))
        return 
    
    try:
        # 파일에서 기본 라인수의 리스트를 효율적으로 읽어와 출력
        init_contents = read_last_n_lines(fpath, lcount=init_lines)
        for line in init_contents:
            print(colorize_log_line(line.strip()))

        # 파일의 끝으로 이동하여 모니터링 시작
        with open(fpath, "r", encoding=cfg.sys_encoding) as file:
            file.seek(0, os.SEEK_END)

            while True:
                new_line = file.readline()
                if new_line:
                    print(colorize_log_line(new_line.strip()))
                else:
                    time.sleep(0.3)

    except KeyboardInterrupt:
        print(msg.exit)
    except Exception as err:
        print(msg.error(err))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("filepath", type=str, nargs="?")
    p.add_argument("-n", "--lines", type=int, default=10)

    args = p.parse_args()
    fpath = cfg.dir / cfg.log_fname if args.filepath is None else Path(args.filepath)
    follow_file(fpath, args.lines)


if __name__ == "__main__":
    main()



