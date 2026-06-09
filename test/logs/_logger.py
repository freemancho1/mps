import logging 
from pathlib import Path 

from . import _logger_config as LOG_CFG

logger = logging.getLogger(LOG_CFG.NAME.upper())

# -- 이 부분이 중요 --
# 중복 로그 생성을 방지하기 위해 이미 핸들러가 설정되어 있다면 추가하지 않음
if not logger.handlers:
    
    # 1. 로거 레벨 설정
    level = getattr(logging, LOG_CFG.LEVEL, LOG_CFG.LVL_INFO)
    logger.setLevel(level)
    
    # 2. 로그 메시지 포멧 정의
    formatter = logging.Formatter(LOG_CFG.FMT_STR, datefmt=LOG_CFG.FMT_DATE)
    
    # 3. 출력 Handler 설정
    if LOG_CFG.OUT_DEVICE == LOG_CFG.OUT_DEVICE_FILE:
        log_path = Path(LOG_CFG.LOG_DIR) / LOG_CFG.LOG_FILE
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
    # 4. 출력 Handler 등록
    logger.addHandler(handler)