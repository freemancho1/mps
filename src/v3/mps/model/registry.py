""" 
모델 레지스트리 ─ 가중치 저장·로드 및 경로 관리 (재현 가능성 원칙 준수)

[재현 가능성]
  - 가중치와 함께 메타 데이터(시드·아키텍처·학습 설정·라벨 분포)를 저장함.
  - "그때는 됐는데 지금은 안된다."를 방지하기 위해, 추론 시 동일 아키텍처로 로드해야 하며,
    메타 데이터로 검증할 수 있음.
"""
from __future__ import annotations 

import torch 
from pathlib import Path 
from typing import Optional, Union

from mps.config import cfg, msg 
from mps.freelibs import logger


def save_checkpoint(
    model: torch.nn.Module, 
    path: Union[Path, str], 
    meta: Optional[dict] = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    save_info = {
        cfg.key.state_dict: model.state_dict(), 
        cfg.key.meta: meta or {}
    }
    logger.debug(msg.training.save_ckpt_info(save_info, path))
    
    torch.save(save_info, path)
    return path 


def load_meta(path: Union[Path, str]) -> dict:
    """ 체크 포인트의 메타 데이터만 조회 (아키텍처 검증용) """
    path = Path(path)
    if not path.exists():
        return {}
    
    # 모델의 가중치를 gpu에 올려 연산용으로 사용하지 않고,
    # 단순히 가중치 메타정보만 확인하기 때문에 cpu에서 처리하는 것이 효과적
    ckpt = torch.load(path, map_location=cfg.key.cpu)
    logger.debug(msg.training.load_ckpt_info(ckpt, path))
    
    return ckpt.get(cfg.key.meta, {}) if isinstance(ckpt, dict) else {}


def weights_exist() -> dict[str, bool]:
    """ Phase-2에서 가중치 존재 여부 확인 (팩토리의 풀백 판단에 사용) """
    return {
        cfg.key.numeric: cfg.path.lstm_model_fpath.exists(),
        cfg.key.pattern: cfg.path.cnn_model_fpath.exists(),
    }
    