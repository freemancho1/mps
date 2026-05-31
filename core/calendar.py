from __future__ import annotations 

from datetime import date, datetime, timedelta 

from mps.config import cfg 


def is_trading_day(dt: date) -> bool:
    """ 영업일 체크: Phase-1에서는 주말 체크만 함 """
    return dt.weekday() < 5

def is_trading_time(dt: datetime) -> bool:
    """ 정규장 시간(일자 포함) 체크 """
    if not is_trading_day(dt.date()):
        return False 
    return cfg.run.open_time <= dt.time() < cfg.run.close_time

def market_open_dt(dt: date) -> datetime:
    return datetime.combine(dt, cfg.run.open_time, tzinfo=cfg.run.timezone)

def market_close_dt(dt: date) -> datetime:
    return datetime.combine(dt, cfg.run.close_time, tzinfo=cfg.run.timezone)

def force_close_dt(
    dt: date, 
    min_force: int = cfg.run.force_close_minutes
) -> datetime:
    """ 강제 청산 기준 시각 ─ 마감 N분전 """
    return market_close_dt(dt) - timedelta(minutes=min_force)

def trading_days(start_date: date, end_date: date) -> list[date]:
    """ 시작일과 종료일 사이의 영업일 목록 (양 끝 포함) """
    result: list[date] = []
    curr_date = start_date 
    while curr_date <= end_date:
        if is_trading_day(curr_date):
            result.append(curr_date)
        curr_date += timedelta(days=1)
    
    return result

def prev_trading_day(dt: date) -> date:
    """ 직전 영업일 찾기 """
    curr_date = dt - timedelta(days=1)
    while not is_trading_day(curr_date):
        curr_date -= timedelta(days=1)
    return curr_date