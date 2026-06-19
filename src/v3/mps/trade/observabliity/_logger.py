""" 
SignalLogger, OrderLogger ─ 모든 신호·주문·체결을 JSONL로 기록

[관측 가능성 원칙]
  - 전략이 왜 진입·청산했는지, 어떤 피처가 신호를 만들었는지, 
    나중에 분석할 수 있어야 함.
  - 모든 신호와 주문을 JSONL 형식으로 저장

[JSONL을 사용한 이유]
  - 한줄 = 한 레코드 → 파일에 append만 하면 됨.
  - cat 명령 등으로 즉시 분석 가능하고 DataFrame으로 전환 편함.
"""
from __future__ import annotations 

import json 
import logging 
from datetime import datetime 
from pathlib import Path 
from typing import Optional, Union

from mps.config import cfg, msg 
from mps.core.types import NumericSignal, PatternSignal, TradeSignal 
from mps.core.types import Order, OrderResult 
from mps.core.libs import serialized


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


class SignalLogger:
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_fname: Optional[str] = None,
    ) -> None:
        log_dir = cfg.path.output if log_dir is None else log_dir
        log_fname = cfg.path.signal_log_fname if log_fname is None else log_fname 
        self._path = log_dir / log_fname 
        self._logger = logging.getLogger(cfg.path.signal_log_title)

    def log(self, signal: Union[NumericSignal, PatternSignal, TradeSignal]) -> None:
        """ 신호를 logging + jsonl 파일 두 곳에 동시 기록 """
        record = {"type": type(signal).__name__, "data": serialized(signal)}
        json_record: str = json.dumps(record, ensure_ascii=False, default=str)

        if cfg.sys.signal_logging_on:
            self._logger.info(json_record)
        with self._path.open("a", encoding="utf-8") as logfile:
            logfile.write(json_record + "\n")


class OrderLogger:
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_fname: Optional[str] = None,
    ) -> None:
        log_dir = cfg.path.output if log_dir is None else log_dir 
        log_fname = cfg.path.order_log_fname if log_fname is None else log_fname 
        self._path = log_dir / log_fname 
        self._logger = logging.getLogger(cfg.path.order_log_title)

    def log(self, order: Order, result: OrderResult) -> None:
        """ 
        주문과 체결 결과를 함께 기록.

        - order: 방향·수량·TakeProfit·StopLoss·만료 시각 포함
        - result: 실제 체결가·수량·슬리피지 포함

        두 객체를 함계 기록해야 '진입 의도 vs 실제 체결' 비교 분석 가능
        """
        record = {"order": serialized(order), "result": serialized(result)}
        json_record: str = json.dumps(record, ensure_ascii=False, default=str)

        if cfg.sys.order_logging_on:
            self._logger.info(json_record)
        with self._path.open("a", encoding="utf-8") as logfile:
            logfile.write(json_record + "\n")