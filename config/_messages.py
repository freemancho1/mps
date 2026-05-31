from mps.core.libs import DictDot
from mps.core.libs import call_function as CF  


tm = DictDot(           # train_model
    title               = "MPS Phase-2 모델 학습",
)

data = DictDot(         # Preprocessing DataIo
    file_not_found      = lambda p: CF(f"Parquet 저장 파일이 존재하지 않습니다: {p}"),
    load_info           = lambda s, e, df: CF(f"불러온 Parquet 파일 정보: 기간({s}~{e}), 크기{df.shape}"),
    result_info         = lambda bars: CF(f"리턴할 봉 리스트 갯 수: {len(bars)}"),
    fetch = DictDot(
        from_kis        = CF("KIS REST API를 이용해 데이터 획득"),
        from_pykrx      = CF("Pykrx 라이브러리를 이용해 데이터 합성"),
        pykrx_info      = lambda s, e, t, d: CF(f"Pykrx 데이터 생성: 기간({s}~{e}), 종목번호: {t}, 데이터 크기: {d.shape}"),
    ),
)