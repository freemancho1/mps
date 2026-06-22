""" 
시장 캘린더 ─ 영업일·정규장 시장·강제청산 시각 계산

[KRX 연동 전 한계]
  - 영업일 판정이 '주말'뿐이라 공휴일(설·추석 등)을 거래일로 오판함.
  - 합성 데이터는 pykrx 실제 거래일 기준으로 생성되므로 백테스트에는 영향이 없지만,
    실시간 전환 전에 KRX 휴장일 캘린더 연동이 반드시 필요
"""
from __future__ import annotations 

from datetime import date, datetime, timedelta 
from typing import Optional 

from mps.core.config import cfg 


def is_trading_day(check_date: date) -> bool:
    """ 영업일 체크 (주말만 제외시킴) """
    return check_date.weekday() < 5

def is_trading_time(check_dt: datetime) -> bool:
    """ 현재 날짜와 시간이 영업 시간인지 판단 """
    if not is_trading_day(check_date=check_dt.date()):
        return False 
    return cfg.market.open_time <= check_dt.time() < cfg.market.close_time

def market_open_dt(check_date: date) -> datetime: 
    """ 점검일 → 점검일+개장시간 반환 """
    return datetime.combine(check_date, cfg.market.open_time, tzinfo=cfg.sys.timezone)

def market_close_dt(check_date: date) -> datetime:
    """ 점검일 → 점검일+패장시간 반환 """
    return datetime.combine(check_date, cfg.market.close_time, tzinfo=cfg.sys.timezone)

def force_close_dt(check_date: date, min_force: Optional[int] = None) -> datetime:
    """ 
    강제청산 기준 시각 = 패장 N(기본값=15)분 전 = date+15:30 - 15 = data+1515
    
    당일 청산 원칙: 이 시각 이후 보유 포지션은 사유 무관 시장가 청산
      - 오버나잇 갭 리스크 완전 차단
    """
    min_force = cfg.market.force_close_minutes if min_force is None else min_force 
    return market_close_dt(check_date) - timedelta(minutes=min_force)

def trading_days(start_date: date, end_date: date) -> list[date]:
    """ 시작일과 종료일 사이 영업일 리스트 (시작·종료일 포함) """
    results: list[date] = []
    
    check_date = start_date 
    while check_date <= end_date:
        if is_trading_day(check_date):
            results.append(check_date)
        check_date += timedelta(days=1)
        
    return results 

def prev_trading_day(base_date: date) -> date:
    """ 기준일 기준 직전 영업일 리턴 """
    check_date = base_date 
    while not is_trading_day(check_date):
        check_date -= timedelta(days=1)
    return check_date