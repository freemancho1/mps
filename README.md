# MPS ─ Muse Pulse System 

KOSPI 단타(데이트레이딩) 자동매매 시스템. <br/>
수치 트랙(LSTML)과 패턴 트랙(1D-CNN)의 합의 신호로 매수 진입하고, 손절·익절·만료·강종 가도르 청산함. 

## 설치
project-home-dir/pyproject.toml에서 `where`부분 수정

```code
[tool.setuptools.packages.find]
where = ["src/v3"]
```

아래 내용을 실행하면, 소스코드를 변경하면서 계속 프로그램을 작성할 수 있도록 설치됨.
* where에 정의된 폴더 아래에 `프로젝트명.egg-info`폴더가 소스코드 정보를 제공함.

```bash
pip install -e .
```

## 실행
```bash
python scripts/run_backtest.py  --ticker 005930 --start 20250101 --end 20251231
python scripts/train_models.py  --ticker 005930 --start 20250101 --end 20251231
```

## 패키지 구조 
#### 소스 코드: src/버전 번호(v1)/프로젝트 명(mps)/
* `core/`: 설정·메시지·타입·인터페이스(port)·유틸리티 ─ 공통부
* `data/`: 분봉 수집·저장(loader·store) + 피처 파이프라인(features) ─ 거래·훈련 공통 데이터 계층
* `modeling/`: 트랙별 모델(numeric·pattern) + 학습·레지스트리·팩토리 관리 ─ 모델 학습부
* `trading/`: 신호 결합·리스크, 집행·관측·백테스트 ─ 거래부 

#### 기타 파일: 버전 밖에서 공통으로 사용하는 폴더
* `artifacts/`: 런타임 산출물(로그·parquet·모델 가중치) ─ 시스템 출력부
  * 파일 구조: artifacts/버전 번호(v1)/logs/ ...
* `documents/`: 프로젝트 설계 원칙·의사결정은 design-philosophy.ipynb 참조
