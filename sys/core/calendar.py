from __future__ import annotations

from datetime import date, time, datetime, timedelta 

from mps.sys import cfg


def is_trading_day(dt: date) -> bool:
    """ 영업일 체크(Phase-1에서는 주말 체크만 함) """
    return dt.weekday() < 5

def is_trading_time(dt: datetime) -> bool:
    """ 정규장 시간(일자 포함) 체크 """
    if not is_trading_day(dt.date()):
        return False 
    t = dt.time()
    return cfg.sys.market_open_time <= t < cfg.sys.market_close_time

def market_open_dt(d: date) -> datetime: 
    return datetime.combine(d, cfg.sys.market_open_time, tzinfo=cfg.sys.timezone)

def market_close_dt(d: date) -> datetime: 
    return datetime.combine(d, cfg.sys.market_close_time, tzinfo=cfg.sys.timezone)

def force_close_dt(
    d: date, 
    minutes_before: int = cfg.sys.force_close_minutes_before
) -> datetime: 
    """ 강제 청산 기준 시각 ─ 마감 N분 전 """
    return market_close_dt(d) - timedelta(minutes=minutes_before)

def trading_days(start: date, end: date) -> list[date]:
    """ 시작~종료일 사이의 거래일 목록 (양 끝 포함). """
    result = []
    curr = start 
    while curr <= end:
        if is_trading_day(curr):
            result.append(curr)
        curr += timedelta(days=1)
    return result 

def prev_trading_day(d: date) -> date: 
    """ 지정 일 직전 영업일 찾기 """
    curr = d - timedelta(days=1)
    while not is_trading_day(curr):
        curr -= timedelta(days=1)
    return curr