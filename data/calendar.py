from __future__ import annotations

from datetime import date, datetime, timedelta

from mps.sys import constants as const

# KST = ZoneInfo("Asia/Seoul")
# _MARKET_OPEN = time(9, 0)
# _MARKET_CLOSE = time(15, 30)


def is_trading_day(dt: date) -> bool:
    return dt.weekday() < 5

def is_trading_time(dt: datetime) -> bool:
    if not is_trading_day(dt.date()):
        return False 
    t = dt.time()
    return const.MARKET_OPEN <= t < const.MARKET_CLOSE

def market_open_dt(d: date) -> datetime:
    return datetime.combine(d, const.MARKET_OPEN, tzinfo=const.KST)

def market_close_dt(d: date) -> datetime:
    return datetime.combine(d, const.MARKET_CLOSE, tzinfo=const.KST)

def force_close_dt(d: date, minutes_before: int = 15) -> datetime:
    """ 마감 N(=15)분 전 강제청산 기준 시간 """
    return market_close_dt(d) - timedelta(minutes=minutes_before)

def trading_days(start_date: date, end_date: date) -> list[date]:
    """ start_date ~ end_date 사이 거래일 목록 리턴 (양 끝 포함). """
    result = []
    curr_date = start_date 
    while curr_date <= end_date:
        if is_trading_day(curr_date):
            result.append(curr_date)
        curr_date += timedelta(days=1)
    return result

def prev_trading_day(d: date) -> date:
    """ 지정한 날짜의 직전 영업일 찾기 """
    curr_date = d - timedelta(days=1)
    while not is_trading_day(curr_date):
        curr_date -= timedelta(days=1)
    return curr_date