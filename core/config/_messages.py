from __future__ import annotations 

from dataclasses import dataclass 
from typing import Callable, TypeAlias

from mps.core.libs import CF 


mfn: TypeAlias = Callable[..., str]     # Message_FuNction, str과 자리수 맞추기 위해 3글자로 함


@dataclass(frozen=True)
class _StoreMessages:
    file_not_found_err      : mfn = lambda path: CF(f"Parquet 파일이 존재하지 않습니다: {path}")
    load_data_info          : mfn = lambda df: CF(f"불러온 데이터 정보: 데이터 크기={df.shape}, 기간={df.index[0]}~{df.index[-1]}") if len(df) > 0 else CF(f"불러온 데이터 정보: 데이터 크기={df.shape}, 기간=없음")
    save_parquet_info       : mfn = lambda df: CF(f"저장한 데이터 정보: 데이터 크기={df.shape}, 기간={df.index[0]}~{df.index[-1]}") if len(df) > 0 else CF(f"저장한 데이터 정보: 데이터 크기={df.shape}, 기간=없음")


@dataclass(frozen=True)
class _LoaderMessages:
    load_result             : mfn = lambda soc, bars: CF(
                                f"데이터 로드 결과: 출처[{soc}], 크기[{len(bars)}], "
                                f"기간: {bars[0].timestamp} ~ {bars[-1].timestamp}"
                                if bars else f"데이터 로드 결과: 출처[{soc}], 크기[0], 기간: 없음"
                            )


@dataclass(frozen=True)
class _FeatureMessages:
    labeling_result         : mfn = lambda bars, label_dist: CF(f"라벨링 대상 분봉 갯 수: {len(bars)}개, 라벨링 결과: {label_dist}")
    ds_window_size          : mfn = lambda s_pit, e_pit: CF(f"데이터셋 학습 윈도우: 시작={s_pit}, 종료={e_pit}")
    invalid_track_err       : mfn = lambda track, track_type: f"데이터셋을 만드는 트랙을 잘 못 설정하였습니다. 입력값: {track}, 기대값: {track_type}"
    too_few_bar_size_err    : mfn = lambda bar_size, win_size: f"정규화를 위한 봉 갯 수가 너무 적습니다. 입력 갯 수: {bar_size}, 기대값: {win_size}"


@dataclass(frozen=True)
class _ModelTrainerMessages: 
    invalid_data_type       : mfn = lambda ds: f"학습용 데이터셋({ds.__class__.__name__}) 오류(Sized 함수 미구현)"
    insufficiend_data       : mfn = lambda ds, base_size: (
                                f"학습용 데이터셋({ds.__class__.__name__}) 크기가 너무 작습니다. "
                                f"입력 데이터 크기: {len(ds)}, 최소 데이터 크기: {base_size}"
                            )
    too_much_embargo        : mfn = lambda e_size, b_size: CF(
                                f"엠바고용 데이터가 너무 많습니다. "
                                f"엠바고용 사이즈: {e_size}, 조정된 사이즈: {b_size}"
                            )
    class_calibration       : mfn = lambda weights: CF(f"클래스 보정 결과: {weights}")
    train_epoch_result      : mfn = lambda epoch, history: CF(
                                f" - epoch[{epoch:02}]:"
                                f" train-loss = {history.train_loss},"
                                f" val-loss = {history.val_loss}, val-acc = {history.val_acc}"
                            )
    train_result            : mfn = lambda model, history: CF(
                                f" - [MODEL TRAIN RESULT] Model: {model.__class__.__name__},"
                                f" Best(Epoch: {history.best_epoch},"
                                f" val-loss: {history.val_loss[history.best_epoch]},"
                                f" val-acc: {history.val_acc[history.best_epoch]})"
                            )

@dataclass(frozen=True)
class _Messages:
    store                   : _StoreMessages = _StoreMessages()
    loader                  : _LoaderMessages = _LoaderMessages()
    feature                 : _FeatureMessages = _FeatureMessages()
    trainer                 : _ModelTrainerMessages = _ModelTrainerMessages()

    logger_like             : mfn = lambda msg: CF(msg)


messages                    : _Messages = _Messages()