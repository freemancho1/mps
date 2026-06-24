""" 정규화 레이어 ─ 수치(롤링 Z-score)·패턴(0~1 정규화) 트랙 정규화 """
from __future__ import annotations 

import numpy as np 
from typing import Optional 

from mps.core.config import cfg, msg 
# TODO types.NumericInput 작업 후
# from mps.core.types import Bar, 