""" 
SignalLogger·OrderLogger ─ 모든 신호·주문·체결을 JSONL로 기록

[관측 가능성 원칙]
  - 전략이 왜 진입·청산했는지, 어떤 피처가 신호를 만들었는지,
    나중에 분석할 수 있어야 함.
    → 모든 신호와 주문을 JSONL(JSON Lines) 형식으로 저장

[파일 구조]
  - logs/
        signals.jsonl   → 모든 TradeSignal 기록
        orders.jsonl    → 모든 주문·체결 결과 기록

[JSONL 선택 이유]
  - 한 줄 = 한 레코드 → 파일에 append만 하면 됨.
  - cat 명령 등으로 즉시 분석 가능하고 DataFrame으로 전환 가능
"""
from __future__ import annotations 

import json 
import logging
from datetime import datetime  
from typing import Optional 
from pathlib import Path 

from mps.sys.core.types import NumericalSignal, PatternSignal, TradeSignal
from mps.sys.core.types import Order, OrderResult
from mps.sys import cfg


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


def _serialized(obj):
    """ datetime 형태의 데이터를 JSON 직렬화가 가능한 dict/str 형태로 변환 """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return {key: _serialized(value) for key, value in vars(obj).items()}
    return obj


class SignalLogger:
    def __init__(
        self, 
        log_dir: Path = cfg.log.dir,
        log_fname: str = cfg.log.signal_log_fname,
    ) -> None:
        self._path = log_dir / log_fname
        self._log = logging.getLogger(cfg.log.signal_str)
        
    def log(self, signal: NumericalSignal | PatternSignal | TradeSignal) -> None:
        """ 
        신호를 Python logging + JSONL 파일 두 곳에 동시 기록
        
        type 필드로 신호 종류를 구분
        data 필드에 신호의 모든 속성 포함 (feature_contrib 포함 → 사후 분석 가능)
        """
        record = {"type": type(signal).__name__, "data": _serialized(signal)}
        self._log.info(json.dumps(record, ensure_ascii=False, default=str))
        self._append(record)
        
    def _append(self, record: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class OrderLogger:
    def __init__(
        self,
        log_dir: Path = cfg.log.dir, 
        log_fname: str = cfg.log.order_log_fname,
    ) -> None:
        self._path = log_dir / log_fname 
        self._log = logging.getLogger(cfg.log.order_str)
        
    def log(self, order: Order, result: OrderResult) -> None:
        """ 
        주문과 체결 결과를 함께 기록
        
        order: 방향·수량·TP/SL·만료 시각 포함
        result: 실제 체결가·수량·슬리피지 포함
        
        두 객체를 함께 기록해야 "진입 의도 vs 실제 체결" 비교 분석 가능.
        """
        record = {"order": _serialized(order), "result": _serialized(result)}
        self._log.info(json.dumps(record, ensure_ascii=False, default=str))
        self._append(record)
        
    def _append(self, record: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")