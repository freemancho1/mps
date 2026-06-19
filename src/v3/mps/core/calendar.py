""" 
시장 캘린더 ─ 영업일·정규장 시장·강제청산 시각 계산

[KRX 연동 전 한계]
  - 영업일 판정이 "주말"뿐이라 공휴일(설·추석 등)을 거래일로 오판함.
  - 합성 데이터는 pykrx 실제 거래일 기준으로 생성되므로 백테스트에는 영향이 없지만,
    실시간 전환 전에 KRX 휴장일 캘린더 연동이 반드시 필요.
"""
from __future__ import annotations 

from datetime import date, datetime, timedelta 
from typing import Optional 

from mps.config import cfg 


def is_trading_day(check_date: date) -> bool:
    """ 영업일 체크 (주말만 제외) """
    return check_date.weekday() < 5

def is_trading_time(check_datetime: datetime) -> bool:
    """ 현재 날짜와 시간이 영업 시간인지 판단 """
    if not is_trading_day(check_datetime.date()):
        return False 
    return cfg.market.open_time <= check_datetime.time() < cfg.market.close_time

def market_open_datetime(check_date: date) -> datetime:
    """ 점검일의 개장 시각 (09:00:00 KST) 반환 """
    return datetime.combine(check_date, cfg.market.open_time, tzinfo=cfg.sys.timezone)

def market_close_datetime(check_date: date) -> datetime:
    """ 점검일의 패장 시각 (15:30:00 KST) 반환 """
    return datetime.combine(check_date, cfg.market.close_time, tzinfo=cfg.sys.timezone)

def force_close_datetime(check_date: date, min_force: Optional[int] = None) -> datetime:
    """ 
    강제청산 기준 시각 = 폐장 N분 전 ─ 기본값 15 = 15:30 - 15분

    당일 청산 원칙: 이 시각 이후 보유 포지선은 사유 무관 시장가 청산 
    ─ 오버나잇 갭 리스크 원천 차단
    """
    if min_force is None:
        min_force = cfg.market.force_close_minutes
    return market_close_datetime(check_date) - timedelta(minutes=min_force)

def trading_days(start_date: date, end_date: date) -> list[date]:
    """ 시작일과 종료일 사이 영업일 리스트 (시작·종료일 포함) """
    results: list[date] = []
    curr_date = start_date 
    while curr_date <= end_date:
        if is_trading_day(curr_date):
            results.append(curr_date)
        curr_date += timedelta(days=1)
    return results 

def prev_trading_day(base_date: date) -> date:
    """ 기준일 기준 직전 영업일 반환 """
    curr_date = base_date - timedelta(days=1)
    while not is_trading_day(curr_date):
        curr_date -= timedelta(days=1)
    return curr_date