""" 
ModelTrainer ─ 학습 기반 모델(LSTM·1D-CNN) 공통 학습 루프

[설계]
  - 두 트랙 모두 3-클래스(BUY·SELL·HOLD) 분류이므로 학습 루프를 공유.
  - 시간 순 train·val 분할(누출 방지, 결정 8 정신) ─ 무작위 셔플 금지.
  - 클래스 불균형(HOLD 다수) 보정을 위해 CrosEntropyLoss에 클래스 가중치 적용.
  - 검증 손실 기준 조기 종료(early stopping)로 과적합 방지
  
[재현 가능성]
  - seed 고정(torch·numpy)으로 동일 데이터·코드·시드 → 동일 결과
"""
from __future__ import annotations 

import torch 
import numpy as np
from typing import Optional 

from mps.config import cfg, msg 
from mps.core.libs import set_seed


# TODO X: 여기 작업 
# "editor.parameterHints.enabled": false,     // 함수 매개변수 힌트 막음
# // "editor.suggestOnTriggerCharacters": false,     // 자동완성 팝업 힌트 막음