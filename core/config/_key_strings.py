from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias 

from mps.core.types import AggregationMode
from mps.core.types import DataSource
from mps.core.types import ExitHoldReason
from mps.core.types import OrderAction, OrderStatus, OrderType
from mps.core.types import PatternSource, PatternName
from mps.core.types import RejectReason


DropKeep: TypeAlias = Literal["first", "last", False]


@dataclass(frozen=True)
class _KeyValues:
    # 알파벳 순
    
    atr_14                  : str = "atr_14"

    bb_lower                : str = "bb_lower"
    bb_mid                  : str = "bb_mid"
    bb_pband                : str = "bb_pband"
    bb_upper                : str = "bb_upper"

    capital                 : str = "--capital"
    class_counts            : str = "class_counts"
    close                   : str = "close"
    count                   : str = "count"
    cpu                     : str = "cpu"
    cuda                    : str = "cuda"

    end                     : str = "--end"
    
    gpu                     : str = "gpu"
    
    high                    : str = "high"
    
    last                    : DropKeep = "last"
    low                     : str = "low"
    
    macd                    : str = "macd"
    macd_diff               : str = "macd_diff"
    macd_signal             : str = "macd_signal"
    max_ms                  : str = "max_ms"
    mean_ms                 : str = "mean_ms"
    meta                    : str = "meta"
    
    numeric                 : str = "numeric"
    
    obv                     : str = "obv"
    open                    : str = "open"
    
    p95_ms                  : str = "p95_ms"
    pattern                 : str = "pattern"
    
    ret_1                   : str = "ret_1"
    ret_20                  : str = "ret_20"
    ret_5                   : str = "ret_5"
    rsi_14                  : str = "rsi_14"

    start                   : str = "--start"
    state_dict              : str = "state_dict"
    
    test_days               : str = "--test_days"
    ticker                  : str = "--ticker"
    timestamp               : str = "timestamp"
    train_days              : str = "--train_days"
    
    volume                  : str = "volume"
    volume_ratio            : str = "volume_ratio"
    
    
@dataclass(frozen=True)
class _StringValue:
    # 알파벳 순

    agg_confluence          : AggregationMode = "CONFLUENCE"        # 합의 요구(AND)
    agg_weighted            : AggregationMode = "WEIGHTED"          # 가중 결합(soft-OR)

    bearish_engulfing       : PatternName = "BEARISH_ENGULFING"
    box_breakout_down       : PatternName = "BOX_BREAKOUT_DOWN"
    box_breakout_up         : PatternName = "BOX_BREAKOUT_UP"
    bullish_engulfing       : PatternName = "BULLISH_ENGULFING"
    buy                     : Literal["BUY"] = "BUY"
    
    cancelled               : OrderStatus = "CANCELLED"
    cnn                     : PatternSource = "CNN"
    cnn_seq                 : PatternName = "CNN_SEQ"               # cnn pattern_name

    daily_loss_limit        : RejectReason = "DAILY_LOSS_LIMIT"

    entry_cutoff            : RejectReason = "ENTRY_CUTOFF"
    evening_star            : PatternName = "EVENING_STAR"

    feature                 : str = "FEATURE"
    filled                  : OrderStatus = "FILLED"
    force_close             : ExitHoldReason = "FORCE_CLOSE"

    hammer                  : PatternName = "HAMMER"
    hold                    : Literal["HOLD"] = "HOLD"

    kis                     : DataSource = "KIS"

    market                  : OrderType = "MARKET"
    morning_star            : PatternName = "MORNING_STAR"

    no_cash                 : RejectReason = "NO_CASH"
    numeric                 : str = "NUMERIC"

    pattern                 : str = "PATTERN"
    pykrx                   : DataSource = "PYKRX"

    rule                    : PatternSource = "RULE"

    sell                    : OrderAction = "SELL"
    shooting_star           : PatternName = "SHOOTING_STAR"
    stop_loss               : ExitHoldReason = "STOP_LOSS"
    store                   : DataSource = "STORE"

    take_profit             : ExitHoldReason = "TAKE_PROFIT"
    time_out                : ExitHoldReason = "TIME_OUT"
    
    vision                  : PatternSource = "VISION"