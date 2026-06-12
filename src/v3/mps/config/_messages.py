from mps.freelibs import DictDot, call_function as CF
from ._config import config as cfg


pp = DictDot(               # Preprocessing
    store = DictDot(
        file_not_found      = lambda path: CF(f"Parquet 파일이 존재하지 않습니다: {path}") ,
        load_store_info     = lambda df: CF(f"불러온 저장 데이터 정보: 날짜 인덱스({df.index[0]}~{df.index[-1]}), 데이터 크기 {df.shape}"),
    ),
)

training = DictDot(
    title                   = f"MPS Phase-{cfg.sys.phase} 모델 학습",
    info                    = lambda title, ticker, s_dt, e_dt: CF(f"{title} 정보: 대상 종목 코드({ticker}), 학습 기간({s_dt} ~ {e_dt})"),
)