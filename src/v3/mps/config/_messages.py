from mps.freelibs import DictDot, call_function as CF
from ._config import config as cfg


pp = DictDot(               # Preprocessing
    store = DictDot(
        file_not_found      = lambda path: CF(f"Parquet 파일이 존재하지 않습니다: {path}") ,
        load_store_info     = lambda df: CF(f"불러온 저장 데이터 정보: 날짜 인덱스({df.index[0]}~{df.index[-1]}), 데이터 크기 {df.shape}"),
        krx_synthetic_info  = lambda df: CF(f"PYKRX 합성 데이터 정보: 날짜 인덱스({df.index[0]}~{df.index[-1]}), 데이터 크기 {df.shape}"),
        load_result         = lambda f, b: CF(f"{f}을 이용해 {len(b)}개의 분봉을 생성함"),
        save_result         = lambda t, df: CF(f"종목코드 [{t}]의 기간({df.index[0]}~{df.index[-1]}) 분봉 {df.shape[0]}개를 저장함."),
    ),
    features = DictDot(
        label_size          = lambda s, l: CF(f"라벨링 대상 분봉 갯 수: {len(s)}개, 라벨링 결과: {l.shape}"),
        dist_labels         = lambda d: CF(f"전체 데이터 라벨링 결과: {d}"),
        dataset_result     = lambda ds, di: CF(f"학습용 샘플 수: {len(ds)}개, 라벨링 분포: {di}"),
        window_range        = lambda s, e: CF(f"데이터셋 학습 윈도우 경계: {s} ~ {e}"),
    ),
)

training = DictDot(
    title                   = f"MPS Phase-{cfg.sys.phase} 모델 학습",
    start                   = CF("학습 시작."),
    info                    = lambda title, ticker, s_dt, e_dt: CF(f"{title} 정보: 대상 종목 코드({ticker}), 학습 기간({s_dt} ~ {e_dt})"),
    model_info              = lambda t, m: CF(f"모델 학습 정보: {t} 트랙(모델명: {m.__class__.__name__})"),
    # load_data_info          = lambda t, s, e: CF(f"학습 데이터 정보: 종목코드({t}), 기간({s}~{e})"),
    not_compute_gradient    = "모델의 gradient가 계산되지 않았습니다.",
    finished                = lambda t: CF(f"학습 완료. 처리 시간: {t}"),
    too_much_embargo        = lambda e, c: CF(f"엠바고용 데이터({e}개)가 너무 많습니다. {c}개로 조정햇습니다."),
    class_calibration       = lambda w: CF(f"클래스 보정 결과: {w}"),
    err = DictDot(
        not_len_func        = lambda ds: f"{ds.__class__.__name__} 오브젝트가 '__len__()' 메서드를 구현하지 않았습니다.",
        insufficient_data   = lambda ds: f"학습에 필요한 데이터가 너무 적습니다. 입력 데이터 크기: {len(ds)}개",
    ),
)