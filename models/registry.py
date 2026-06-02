""" 
모델 레지스트리 ─ 가중치 저장·로드 및 경로 관리 (재현 가능성 원칙)

[재현 가능성 (README 변경 불가 원칙)]
  - 가중치와 함께 메타 데이터(시드·아키텍처·학습 설정·라벨 분포)를 저장함.
  - "그때는 됐는데 지금은 안된다."를 방지하기 위해, 추론 시 동일 아키텍처로
    로드해야 하며 메타 데이터로 검증할 수 있음.
"""
from __future__ import annotations 

import torch
from pathlib import Path 
from typing import Optional, Any

from mps.config import cfg, msg 


def save_checkpoint(
    model: torch.nn.Module, 
    path: Path, 
    meta: Optional[dict] = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    save_info = {"state_dict": model.state_dict(), "meta": meta or {}}
    print(msg.trading.save_model_info(save_info))
    torch.save(save_info, path)
    
    return path


def load_meta(path: Path) -> dict:
    """ 체크 포인트의 메타 데이터만 조회 (아키텍처 검증용) """
    path = Path(path)
    if not path.exists():
        return {}
    
    # 모델의 가중치를 gpu에 올려 연산용으로 사용하지 않고,
    # 단순히 가중치 메타정보만 확인하기 위해서는 cpu에서 처리하는것이 효과적
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}


def weights_exist() -> dict[str, bool]:
    """ Phase-2 가중치 존재 여부 확인 (팩토리의 풀백 판단에 사용) """
    return {
        "numeric": cfg.model.lstm_model_fpath.exists(),
        "pattern": cfg.model.cnn_model_fpath.exists(),
    }
    


