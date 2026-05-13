from datetime import time 
from zoneinfo import ZoneInfo

# System
SEED = 42

# TimeZone
KST = ZoneInfo("Asia/Seoul")
DATE_FORMAT = "%Y%m%d"

# Market Info
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(15, 30)
MINUTES_PER_DAY = 390           # 09:00~15:30 = 390분