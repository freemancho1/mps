# MPS ─ Muse Pulse System 

KOSPI 단타(데이트레이딩) 자동매매 시스템. <br/>
수치 트랙(LSTML)과 패턴 트랙(1D-CNN)의 합의 신호로 매수 진입하고, 손절·익절·만료·강종 가도르 청산함. 

<!-- ## 설치
project-home-dir/pyproject.toml에서 `where`부분 수정

```code
[tool.setuptools.packages.find]
where = ["src/v3"]
```

아래 내용을 실행하면, 소스코드를 변경하면서 계속 프로그램을 작성할 수 있도록 설치됨.
* where에 정의된 폴더 아래에 `프로젝트명.egg-info`폴더가 소스코드 정보를 제공함.

```bash
pip install -e .
``` -->

## 실행
```bash
cd ~/projects/mps
conda activate mpsdev

# 간략 스크립트 실행 (버전을 입력하지 않으면 기본값 v3임)
./run train_models.py 
./run train_models.py v3
./run backtest.py v4

# 정상적인 방법
cd src/v3/mps
python scripts/backtest.py  --ticker 005930 --start 20250101 --end 20251231
python scripts/train_models.py  --ticker 005930 --start 20250101 --end 20251231

# 모니터링
cd ~/projects/mps
./monitoring
```

## 패키지 구조 

### mps
#### document:: docs ─ mps.docs
* `documents/`: 프로젝트 설계 원칙·의사결정은 design-philosophy.ipynb 참조

#### source:: 
* `core/`: config·타입·인터페이스(port)·유틸리티 ─ 공통부
* `data/`: 분봉 수집·저장(loader·store) + 피처 파이프라인(features) ─ 거래·훈련 공통 데이터 계층
* `model/`: 트랙별 모델(numeric·pattern) + 학습·레지스트리·팩토리 관리 ─ 모델 학습부
* `trade/`: 실행, 백테스트, 신호, 관측 등 거래의 모든 내용이 들어감
* `scripts/`: 각종 실행 스크립트, train_models, backtest.py
* `x.test/`: 각종 태스트 프로그램

#### artifacts:: artifacts ─ msp/artifacts
* `models/`: 모델 학습 결과 저장
* `monitoring/`: 시스템이 관리하는 로그 저장(통계 작성용)
* `store/`: 분봉 파일 저장용 폴더
