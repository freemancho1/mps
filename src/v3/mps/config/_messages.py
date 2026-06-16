from __future__ import annotations 

from mps.freelibs import DictDot, call_function as CF


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
    title                   = f"MPS Phase-2 모델 학습",
    start                   = CF("학습 시작."),
    info                    = lambda title, ticker, s_dt, e_dt: CF(f"{title} 정보: 대상 종목 코드({ticker}), 학습 기간({s_dt} ~ {e_dt})"),
    model_info              = lambda t, m: CF(f"모델 학습 정보: {t} 트랙(모델명: {m.__class__.__name__})"),
    # load_data_info          = lambda t, s, e: CF(f"학습 데이터 정보: 종목코드({t}), 기간({s}~{e})"),
    not_compute_gradient    = "모델의 gradient가 계산되지 않았습니다.",
    finished                = lambda t: CF(f"학습 완료. 처리 시간: {t}"),
    too_much_embargo        = lambda e, c: CF(f"엠바고용 데이터({e}개)가 너무 많습니다. {c}개로 조정햇습니다."),
    class_calibration       = lambda w: CF(f"클래스 보정 결과: {w}"),

    epoch_result            = lambda e, h: CF(f" - epoch[{e:02}]: train-loss = {h.train_loss[-1]:.4f}, val-loss = {h.val_loss[-1]:.4f}, val-acc = {h.val_acc[-1]:.4f}"),
    result                  = lambda m, h: CF(f"{m.__class__.__name__} 모델 학습 결과: val_loss={h.val_loss[h.best_epoch]:.4f}, val_acc={h.val_acc[h.best_epoch]:.4f}"),

    # 저장·불러온 체크포인트 정보
    save_ckpt_info          = lambda i, p: CF(f"저장된 체크포인트 정보: {i.keys()}, 저장 위치: {p}"),
    load_ckpt_info          = lambda i, p: CF(f"불러온 체크포인트 정보: {i.keys()}, 저장 위치: {p}"),
    
    err = DictDot(
        not_len_func        = lambda ds: f"{ds.__class__.__name__} 오브젝트가 '__len__()' 메서드를 구현하지 않았습니다.",
        insufficient_data   = lambda ds: f"학습에 필요한 데이터가 너무 적습니다. 입력 데이터 크기: {len(ds)}개",
    ),
)

bt = DictDot(               # BackTest
    script_title            = f"MPS Phase-2 백테스트 실행 스크립트",
    title                   = lambda a: CF(f"MPS Phase-2 백테스트 실행: 입력값={a}"),
    result                  = lambda t: CF(f"MPS Phase-2 백테스트 종료: 전체 처리시간={t}"),
    wf_info                 = lambda s: CF(f"WarkForward Validator 실행 정보: 훈련일자={s._train_days}일, 시험일자={s._test_days}일, 초기자본={s._capital:,.0f}원"),
    wf_fold_info            = lambda f, tr, w, te: CF(f" - Fold[{f:03}]: train({tr[0]}~{tr[-1]}), warmup({w[0]}~{w[-1]}), test({te[0]}~{te[-1]})"),
    wf_fold_train_result    = lambda m: CF(f"   = {m.__class__.__name__}, Result: {m.state_dict()}"),
    
    err = DictDot(
        no_data             = CF("학습에 사용할 데이터가 존재하지 않습니다."),  
        insufficient_data   = lambda e, f: CF(f"학습에 필요한 데이터가 충분하지 않아 이 폴드({f})는 건너뜀니다. [ERROR] {str(e)}"),
    ),
)