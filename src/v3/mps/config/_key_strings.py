from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mps.core.types import ExitHoldReason
from mps.core.types import OrderAction, OrderStatus, OrderType
from mps.core.types import PatternSource
from mps.core.types import SignalDirection


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

    end                     : str = "--end"
    
    high                    : str = "high"
    
    low                     : str = "low"
    
    macd                    : str = "macd"
    macd_diff               : str = "macd_diff"
    macd_signal             : str = "macd_signal"
    
    obv                     : str = "obv"
    open                    : str = "open"
    
    ret_1                   : str = "ret_1"
    ret_20                  : str = "ret_20"
    ret_5                   : str = "ret_5"
    rsi_14                  : str = "rsi_14"

    start                   : str = "--start"
    state_dict              : str = "state_dict"
    
    ticker                  : str = "--ticker"
    
    volume                  : str = "volume"
    volume_ratio            : str = "volume_ratio"
    
    
@dataclass(frozen=True)
class _StringValue:
    # 알파벳 순

    bearish_engulfing       : str = "BEARISH_ENGULFING"
    box_breakout_down       : str = "BOX_BREAKOUT_DOWN"
    box_breakout_up         : str = "BOX_BREAKOUT_UP"
    bullish_engulfing       : str = "BULLISH_ENGULFING"
    buy                     : Literal["BUY"] = "BUY"
    
    cancelled               : OrderStatus = "CANCELLED"

    evening_star            : str = "EVENING_STAR"

    filled                  : OrderStatus = "FILLED"
    force_close             : ExitHoldReason = "FORCE_CLOSE"

    hammer                  : str = "HAMMER"
    hold                    : Literal["HOLD"] = "HOLD"

    kis                     : str = "KIS"

    market                  : OrderType = "MARKET"
    morning_star            : str = "MORNING_STAR"

    pykrx                   : str = "PYKRX"

    rule                    : PatternSource = "RULE"

    sell                    : OrderAction = "SELL"
    shooting_star           : str = "SHOOTING_STAR"
    stop_loss               : ExitHoldReason = "STOP_LOSS"
    store                   : str = "STORE"

    take_profit             : ExitHoldReason = "TAKE_PROFIT"
    time_out                : ExitHoldReason = "TIME_OUT"