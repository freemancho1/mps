from mps.core.libs import DictDot
from mps.core.libs import call_function as CF  


tm = DictDot(           # train_model
    title               = "MPS Phase-2 모델 학습",
    info                = lambda t, s, e: CF(f"모델 학습 대상 종목: {t}, 기간: {s}~{e}"),
)

bt = DictDot(           # BackTest
    title               = "MPS Phase-2 백테스트",
    args_info           = lambda a: CF(f"args info: {a}"),
    dataload_info       = lambda lf, b: CF(f"[{lf}]에서 데이터 로드: {len(b):,}개"),
    dataload_err        = CF("학습에 사용할 데이터가 없습니다."),
)

data = DictDot(         # Preprocessing DataIo
    file_not_found      = lambda p: CF(f"Parquet 저장 파일이 존재하지 않습니다: {p}"),
    load_info           = lambda s, e, df: CF(f"불러온 Parquet 파일 정보: 기간({s}~{e}), 크기{df.shape}"),
    result_info         = lambda bars: CF(f"리턴할 봉 리스트 갯 수: {len(bars)}"),
    fetch_result        = lambda load, bars: CF(f"{load}를 이용해 {len(bars)}개의 분봉을 생성함."),
    fetch_pykrx_result  = lambda s, e, t, d: CF(f"Pykrx 데이터 생성: 기간({s}~{e}), 종목번호: {t}, 데이터 크기: {d.shape}"),
)

features = DictDot(     # Preprocessing Features
    label_size          = lambda num: CF(f"라벨링할 봉 갯 수: {num:,}개"),
)

training = DictDot(
    not_compute_gradient = "gradient가 계산되지 않았습니다.",
    track_title         = lambda b, t, m, p: CF(
                            f"===== [{t.upper()}] ===============================\n"
                            f"학습용 봉 갯 수: {len(b):,}개, 모델 명: {m.__class__.__name__}, 저장 위치: {p}"
                        ),
    sample_labels       = lambda ds, dist: CF(f"학습 샘플: {len(ds)}개, 라벨 분포: {dist}"),
    save_model_info     = lambda d: CF(f"model save info: {d}"),
    result_epoch        = lambda e, tl, vl, va: CF(f" - epoch: {e}, train loss = {tl:.4f}, val loss = {vl:.4f}, val acc = {va:.4f}"),
    result              = lambda h: CF(f"학습 결과: 최적 에폭({h.best_epoch}), val_loss={h.best_val_loss:.4f}, val_acc={h.val_acc[h.best_epoch]:.3f}"),
    finished            = lambda t: CF(f"학습 완료. 처리시간: {t}"),
    err = DictDot(
        not_len_func    = "데이터셋이 __len__ 메서드를 구현하지 않았습니다.",
        dataset_size    = lambda ds: f"학습 샘플 부족: {len(ds)}개 (최소 10개 필요)",
    ),
)