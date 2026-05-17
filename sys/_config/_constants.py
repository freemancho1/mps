# ── TEST DATA ────────────────────────────
TEST_TICKERS                = ["005930"]
TEST_START_DATE             = "2025-01-01"
TEST_END_DATE               = "2025-12-31"
TEST_CAPITAL                = 10_000_000.0
LOOKBACK_MINUTES            = 120
FORCE_CLOSE_MINUTES_BEFORE  = 15

COMMISSION_RATE             = 0.00015       # 증권사 수수료(편도 0.015%)
TAX_RATE                    = 0.0018        # 코스피 매도 세율 (0.18%)
SLIPPAGE_RATE               = 0.001         # 슬리피지 (보수적으로 0.1%)

# ── SYSTEM ──────────────────────────────
# common
SEED                        = 42

# directory
DATA_DIR                    = "data"
STORE_DIR                   = "store"
LOG_DIR                     = "logs"
SNAPSHOTS_DIR               = "snapshots"

# date & time
CURR_TIMEZONE               = "Asia/Seoul"
DATE_FORMAT                 = "%Y%m%d"

# ── MARKET INFO ────────────────────────────
MARKET_OPEN_TIME            = "09:00"
MARKET_CLOSE_TIME           = "15:30"
MINUTES_PER_DAY             = 60 * 6 + 30   # 09:00 ~ 15:30 = 390분