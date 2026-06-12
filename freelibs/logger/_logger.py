import logging 
from pathlib import Path 

from ._settings import log_config as cfg


logger = logging.getLogger(cfg.name.upper())


# 중복 로그 생성을 방지하기 위해 이미 핸들러가 설정되어 있다면 추가하지 않음
if not logger.handlers:
    
    # ── 1. 로커 레벨 설정 ──────────────────
    level = getattr(logging, cfg.level, cfg.info)
    logger.setLevel(level)

    # ── 2. 로그 메시지 포멧 정의 ───────────────
    formatter = logging.Formatter(cfg.string_format, datefmt=cfg.datetime_format)

    # ── 3. 출력 Handler 설정 ─────────────────
    if cfg.out_device == cfg.out_device_file:
        fpath = cfg.dir / cfg.log_fname
        handler = logging.FileHandler(fpath, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

    # ── 4. 출력 Handler 등록 ─────────────────
    logger.addHandler(handler)