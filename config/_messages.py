from mps.core.libs import DictDot
from mps.core.libs import call_function as CF  


tm = DictDot(           # train_model
    title               = "MPS Phase-2 모델 학습",
    info                = lambda t, s, e: CF(f"모델 학습 대상 종목: {t}, 기간: {s}~{e}"),
)

data = DictDot(         # Preprocessing DataIo
    file_not_found      = lambda p: CF(f"Parquet 저장 파일이 존재하지 않습니다: {p}"),
    load_info           = lambda s, e, df: CF(f"불러온 Parquet 파일 정보: 기간({s}~{e}), 크기{df.shape}"),
    result_info         = lambda bars: CF(f"리턴할 봉 리스트 갯 수: {len(bars)}"),
    fetch_result        = lambda load, bars: CF(f"{load}를 이용해 {len(bars)}개의 분봉을 생성함."),
    fetch_pykrx_result  = lambda s, e, t, d: CF(f"Pykrx 데이터 생성: 기간({s}~{e}), 종목번호: {t}, 데이터 크기: {d.shape}"),
)

features = DictDot(     # Preprocessing Features
    label_size          = lambda num: CF(f"라벨링할 봉 갯 수: {num}"),
)

trading = DictDot(
    not_compute_gradient = "gradient가 계산되지 않았습니다.",
)