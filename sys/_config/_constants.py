# ── TEST DATA ────────────────────────────
TEST_TICKERS                = ["005930"]
TEST_START_DATE             = "20250101"
TEST_END_DATE               = "20251231"
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
STORE_FNAME                 = "minute_bars.parquet"
LOG_DIR                     = "logs"
SNAPSHOTS_DIR               = "snapshots"

# date & time
CURR_TIMEZONE               = "Asia/Seoul"
DATE_FORMAT                 = "%Y%m%d"

# KIS API Config
VAR_KIS_APP_KEY             = "KIS_APP_KEY"
VAR_KIS_APP_SECRET          = "KIS_APP_SECRET"
VAR_KIS_ACCOUNT_NO          = "KIS_ACCOUNT_NO"
VAR_KIS_MOCK                = "KIS_MOCK"

# ── MARKET INFO ────────────────────────────
MARKET_OPEN_TIME            = "09:00"
MARKET_CLOSE_TIME           = "15:30"
MINUTES_PER_DAY             = 60 * 6 + 30   # 09:00 ~ 15:30 = 390분