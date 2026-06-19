from __future__ import annotations 

import os 
import math 
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo 
from dataclasses import dataclass, field, asdict 

from ._key_strings import _KeyValues, _StringValue
from mps.core.types import SignalDirection, PatternName


# ─────────────────────────────────────
#   프로그램 실행 정보
# ─────────────────────────────────────

@dataclass(frozen=True)
class _RunConfig:
    """ 프로젝트 수행 관련 설정값 정의 """
    
    # 훈련 종목
    tickers                 : tuple[str, ...] = ("005930",)

    # 시작·종료 일자
    start_date_str          : str           = "20250101"
    end_date_str            : str           = "20251231"

    # 초기 자본금
    init_capital            : float         = 10_000_000.0
    
    # 백테스트 관련 윈도우 정보
    train_days              : int           = 30        # walk_forward 폴드별 학습 윈도우 일자
    test_days               : int           = 10        # walk_forward 테스트 윈도우 크기(2주)


# ─────────────────────────────────────
#   시스템 + 경로 정보 
# ─────────────────────────────────────

@dataclass(frozen=True)
class _SystemConfig:
    name                    : str           = "MPS"
    phase                   : int           = 2
    version                 : str           = "v3"
    
    seed                    : int           = 42
    
    zero                    : float         = 1e-8
    
    timezone                : ZoneInfo      = field(default_factory=lambda: ZoneInfo("Asia/Seoul"))
    date_format             : str           = "%Y%m%d"
    time_format             : str           = "%H:%M:%S"
    datetime_format         : str           = "%Y%m%d_%H%M%S"

    signal_logging_on       : bool          = True
    order_logging_on        : bool          = True


@dataclass(frozen=True)
class _PathConfig:
    # version root: ~/projects/mps/src/v3
    # ─ 이 아래에 mps(소스코드 루트), artifacts(산출물 루트)로 분리해 관리
    _root_dir               : Path          = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    
    # 저장소·로그 파일명
    store_fname             : str           = "minute_bars.parquet"
    signal_log_title        : str           = "signal"
    signal_log_fname        : str           = f"{signal_log_title}s.jsonl"
    order_log_title         : str           = "order"
    order_log_fname         : str           = f"{order_log_title}s.jsonl"

    @property 
    def _artifacts_dir(self) -> Path:
        return self._root_dir / "artifacts"
    
    # @property 
    # def monitoring(self) -> Path:
    #     # 모니터링 로그
    #     return self._artifacts_dir / "monitoring"

    @property 
    def output(self) -> Path:
        # output logger(signal, order) 디렉토리(logs)와 구분해 저장
        return self._artifacts_dir / "output"       
    
    @property 
    def store(self) -> Path:
        return self._artifacts_dir / "store"
    
    @property 
    def models(self) -> Path:
        return self._artifacts_dir / "models"
    
    @property
    def lstm_model_fpath(self) -> Path: 
        return self.models / "lstm_numeric.pt"
    
    @property 
    def cnn_model_fpath(self) -> Path:
        return self.models / "cnn_numeric.pt"
    
    def ensure_dirs(self) -> None:
        """ 산출물 관련 디렉토리 일괄 생성 ─ 설정파일 생성 시 1회(명시적으로) 실행 """
        for d in (self.output, self.store, self.models):
            d.mkdir(parents=True, exist_ok=True)


    # _base_dir               : Path          = field(repr=False)

    # store                   : Path          = field(init=False)
    # store_fname             : str           = "minute_bars.parquet"
    
    # logs                    : Path          = field(init=False)
    # signal_log_fname        : str           = "signals.jsonl"
    # order_log_fname         : str           = "orders.jsonl"

    # def __post_init__(self) -> None:
    #     self.store = self._base_dir / "store"
    #     self.logs = self._base_dir / "logs"
        
    #     # 현재 객체의 field중에 Path 속성을 가진 필드를 찾아서
    #     # 해당 디렉토리가 없으면 해당 디렉토리를 생성함.
    #     for f in fields(self):
    #         attr = getattr(self, f.name)
    #         if isinstance(attr, Path):
    #             attr.mkdir(parents=True, exist_ok=True)


# KIS API 연계 정보 
@dataclass(frozen=True)
class _KisApiConfig:
    
    @property 
    def app_key(self) -> str:
        return os.environ.get("KIS_APP_KEY", "")
    
    @property 
    def app_secret(self) -> str: 
        return os.environ.get("KIS_APP_SECRET", "")
    
    @property 
    def app_account_no(self) -> str:
        return os.environ.get("KIS_ACCOUNT_NO", "")
    
    @property
    def mock(self) -> bool:
        return os.environ.get("KIS_MOCK", "true").lower() == "true"


# ─────────────────────────────────────
#   시장 상수 ─ 한국 시장 기본 정보
# ─────────────────────────────────────

@dataclass(frozen=True)
class _MarketConfig:
    open_time_str           : str           = "09:00"       
    close_time_str          : str           = "15:30"       
    days_per_year           : int           = 242           # 연 영업일 개략값 (25년 242일) 
    minutes_per_day         : int           = 60 * 6 + 30   # = 390
    force_close_minutes     : int           = 15            # 마감 N분 전 강제 청산 (15:15)

    @property 
    def open_time(self) -> time:
        return time(int(self.open_time_str[:2]), int(self.open_time_str[3:]))
    
    @property 
    def close_time(self) -> time:
        return time(int(self.close_time_str[:2]), int(self.close_time_str[3:]))
    
    @property 
    def bars_per_day(self) -> int:
        # 일분봉은 일 분 수와 동일 ─ 5분봉 등으로 전환 시 이 부분만 수정하면 됨
        return self.minutes_per_day                         


# ─────────────────────────────────────
#   모델 훈련 설정 ─ LSTM, CNN, Parameter
# ─────────────────────────────────────

@dataclass(frozen=True)
class _LSTMConfig:
    input_size              : int           = 14
    hidden_size             : int           = 64
    num_layers              : int           = 2
    num_classes             : int           = 2     # BUY·HOLD 사냐마냐
    dropout                 : float         = 0.2
    
    def to_dict(self) -> dict:
        return asdict(self)
    

@dataclass(frozen=True)
class _CNNConfig:
    in_channels             : int           = 5
    num_classes             : int           = 2
    dropout                 : float         = 0.2
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    
@dataclass(frozen=True)
class _HyperparameterConfig:
    epochs                  : int           = 50
    batch_size              : int           = 64
    lr                      : float         = 1e-4
    weight_decay            : float         = 1e-4
    val_ratio               : float         = 0.2       # 뒤 20%를 가지고 검증
    patience                : int           = 10        # 조기 종료를 위한 검사 횟 수    
    

@dataclass(frozen=True)
class _TrainConfig:
    lstm_settings           : _LSTMConfig   = field(default_factory=_LSTMConfig)
    cnn_settings            : _CNNConfig    = field(default_factory=_CNNConfig)

    # 하이퍼파라메터
    hyper_params            : _HyperparameterConfig = field(default_factory=_HyperparameterConfig)


@dataclass(frozen=True)
class _ModelConfig:
    torch_device            : str           = "cuda"
    numeric_track           : str           = "numeric"
    pattern_track           : str           = "pattern"
    default_track           : str           = numeric_track

    min_dataset_size        : int           = 100

    # 14개 피처 ─ 순서가 생성되는 행렬 컬럼이므로 변경하면 안됨.
    # (불변 보장을 위해 tuple 사용)
    feature_names           : tuple[str, ...] = (
                                                "rsi_14",
                                                "macd", "macd_signal", "macd_diff",
                                                "bb_upper", "bb_mid", "bb_lower", "bb_pband",
                                                "obv", "atr_14", "volume_ratio",
                                                "ret_1", "ret_5", "ret_20"
                                            )
    
    @property 
    def feature_idx(self) -> dict[str, int]:
        """ 피처명 → 컬럼 인덱스. ThresholdModel의 임계값 판정 등에 사용 """
        return {name: idx for idx, name in enumerate(self.feature_names)}
    
    @property
    def feature_count(self) -> int:
        return len(self.feature_names)
    

# ─────────────────────────────────────
#   거래 관련 상수
# ─────────────────────────────────────

@dataclass(frozen=True)
class _BarrierConfig:
    """ 
    라벨러(학습)와 ExitPolicy(청산)가 같은 값을 공유해야
    "모델이 학습한 것"과 "시스템이 실행하는 것"이 일치함.

    비대챙(+2%/-1%) 이유: 손실을 이익보다 빨리 차단해 덜 손해보고 하기 위함
    """
    take_profit             : float         = 0.02      # +2%
    stop_loss               : float         = 0.01      # -1%
    time_horizon            : int           = 60        # 60분 


@dataclass(frozen=True)
class _CostConfig:
    commission_rate         : float         = 0.00015   # 위탁 수수료 (편도 0.015%)
    tax_rate                : float         = 0.0018    # 증권 거래세 (매도 시 0.18%)
    slippage_rate           : float         = 0.001     # 슬리피지 (편도 0.1%)
    roundtrip_rate          : float         = (commission_rate + slippage_rate) * 2 + tax_rate  # 0.41%


@dataclass(frozen=True)
class _SignalConfig:
    # ── Phase-1 룰 모델(threshold) 파라미터 ───────────────
    rsi_oversold            : float         = 35.0      # RSI 과매도 경계(이하면 BUY 후보)
    rsi_overbought          : float         = 65.0      # RSI 과매수 경계(롱 온리에선 미사용)
    rsi_closeover_base      : float         = 0.3       # 크로스오버 베이스 신뢰도

    # ── 두 트랙 결합 ──────────────────────────
    numeric_weight          : float         = 0.5       # 수치트랙 결합 가중치
    pattern_weight          : float         = 0.5       # 패턴트랙 결합 가중치
    require_confluence      : bool          = True      # 두 트랙 모두 BUY인 경우에만 진입

    # ── 필터 값 ─────────────────────────────
    max_latency_ms          : float         = 5000.0    # 합산 지연 한도

    # [수익성-C] 손익분기 신뢰도 위에 얹는 안전 마진
    #   - 최종 임계값 min_combined_score에서 
    #     (sl+왕복비용)/(tp+sl)+score_margin으로 산출 = 0.550
    score_margin            : float         = 0.08

    # ── 무신호 기본값 ──────────────────────────
    # tuple은 불변이라 직접 기본값 처리할 수 있지만,
    # 내부 dict가 공유될 수 있기 때문에 factory로 감쌈.
    no_signal_numeric       : tuple[SignalDirection, float, dict] = field(
                                                default_factory=lambda: ("HOLD", 0.0, {})
                                            )
    no_signal_pattern       : tuple[SignalDirection, float, PatternName] = ("HOLD", 0.0, "NONE")
    

@dataclass(frozen=True)
class _PatternConfig:
    single_confidence       : float         = 0.6       # 단봉
    double_confidence       : float         = 0.7       # 이중봉
    triple_confidence       : float         = 0.75      # 삼봉
    chart_confidence        : float         = 0.65      # 차트
    
    chart_min_size          : int           = 21
    

@dataclass(frozen=True)
class _RiskConfig:
    max_capital_pct         : float         = 0.1       # 1회 투입 상한 = 초기 자본의 10%

    # [수익성-B] 진입 컷오프: 강제청산(15:15) N분 전부터 신규 진입 금지.
    #   - 마감 임박 진입은 보유 가능 시간이 짧아 익절 확률은 낮고,
    #     왕복 비용은 확정(=0.41%)으로 구조적으로 손실 거래일 가능성이 높음
    entry_cutoff_minutes    : int           = 30

    # [수익성-E] 브레이크이븐 스톱: 미실현 + trigger 도달 시 "다음 봉부터"
    #   - 손절선을 진입가 * (1 + buffer)로 상향
    #   - buffer(+0.5%) > 왕복비용(0.41%) → 되밀려도 본전 이상으로 마감
    use_breakeven_stop      : bool          = True 
    breakeven_trigger       : float         = 0.01      # +1.0% 도달 시 발동
    breakeven_buffer        : float         = 0.005     # 스톱을 진입가 +0.5%로 지정

    # [수익성-F] 일일 손실 한도: 당일 실현 손실이 자본의 -1%를 넘으면
    #            그날 신규 진입 중단 (레짐 급변 시 연속 손절 차단).
    daily_loss_limit_pct    : float         = 0.01

    # [수익성-D] 신뢰도 비례 사이증: score를 [임계값, 1.0] → 배율 [0.7, 1.3] 선형 매핑
    #   - Kelly(Phase-3) 이전의 보수적 "확신 비례 베팅"
    conviction_sizing       : bool          = True 
    conviction_min_factor   : float         = 0.7
    conviction_max_factor   : float         = 1.3
 

@dataclass(frozen=True)
class _TradeConfig:
    barrier                 : _BarrierConfig= field(default_factory=_BarrierConfig)
    cost                    : _CostConfig   = field(default_factory=_CostConfig)
    signal                  : _SignalConfig = field(default_factory=_SignalConfig)
    risk                    : _RiskConfig   = field(default_factory=_RiskConfig)
    pc                      : _PatternConfig= field(default_factory=_PatternConfig)

    @property 
    def breakeven_confidence(self) -> float:
        """ 
        [수익성-C] 손익분기 신뢰도 ─ 기대값이 0이 되믄 BUY 확률 p*

        TakeProfit/StopLoss 두 결과만 가정한 단순 기대값 모델:
          - EV = p·TP - (1-p)·SL - 왕복비용 = 0
            → p* = (SL + 비용) / (TP + SL)
          - 기본 파라미터(2% / 1% / 0.41%)에서 p* = 0.470
        """
        return (self.barrier.stop_loss + self.cost.roundtrip_rate) \
             / (self.barrier.take_profit + self.barrier.stop_loss)
    
    @property 
    def min_combined_score(self) -> float:
        """ 
        신호 통과 임계값 = 손익분기 신뢰도 + 안전 마진 = 0.470 + 0.08 = 0.550

        - TP/SL/비용을 변경하면 임계값이 자동으로 재계산됨.
        """
        return self.breakeven_confidence + self.signal.score_margin
    

# ─────────────────────────────────────
#   데이터 상수 ─ 룩백·합성 데이터 등 데이터 관련 상수
# ─────────────────────────────────────

@dataclass(frozen=True)
class _DataConfig:
    # 캐시 데이터를 이용하지 않고 강제로 다시 데이터를 읽어오는 옵션
    force_refresh_data      : bool          = False 

    # 룩백: 신호 생성 전 반드시 확보할 과거 봉 수 (120~240분(봉))
    lookback_minutes        : int           = 120 
    # 지표 워밍업 봉 수 ─ 시뮬레이터 버퍼 = lookback + warmup_bars
    #   - MACD(EMA26+Signal9)=35봉, BB·ret_20=20봉 → 50봉이면 EMA 수렴에 충분
    #   - 버퍼 앞부분 NaN → 0 구간이 룩백 윈도우에 섞이는 것을 방지하는 역할
    warmup_bars             : int           = 50

    # pykrx 일봉 생성 시 옵션
    volume_weight_09        : float         = 2.6   # 개장 30분 거래량 비율
    volume_weight_12        : float         = 1.5   # 점심시간 거래량 비율
    volume_weight_15        : float         = 2.3   # 폐장시 거래량 비율
    min_max_noise           : float         = 0.5   # 고·저가 노이즈 비율

    # Direction To Index or Index To Direction
    dir2idx                 : dict[SignalDirection, int] = field(
                                                default_factory=lambda: {"BUY": 0, "HOLD": 1}
                                            )
    idx2dir                 : dict[int, SignalDirection] = field(
                                                default_factory=lambda: {0: "BUY", 1: "HOLD"}
                                            )

    # ── 교차 파생값 정의 ────────────────────────
    _market                 : _MarketConfig = field(default_factory=_MarketConfig)
    _barrier                : _BarrierConfig= field(default_factory=_BarrierConfig)

    @property 
    def buffer_days(self) -> int:
        """ 룩백 충족에 필요한 워밍업 "거래일" 수 (walk-forward 버퍼) """
        return math.ceil(self.lookback_minutes / self._market.minutes_per_day) + 1
    
    @property 
    def buffer_bars(self) -> int:
        """ 시뮬레이터 deque 버퍼 크기 = 룩백 + 지표 워밍업 봉 수. """
        return self.lookback_minutes + self.warmup_bars
    
    @property 
    def embargo_bars(self) -> int:
        """ 
        train/val 경계 엠바고(봉 수) = lookback + horizon = 120 + 60 = 180 
        
        train 마지막 샘플의 "라벨"은 horizon분 뒤 봉까지, 
        val 첫 샘플의 "윈도우"는 lookback분 앞 봉까지 보므로(참조하므로),
        이 부분은 비워야 같은 봉을 train과 val에서 공유하는 누수가 사라짐.
        """
        return self.lookback_minutes + self._barrier.time_horizon


# ─────────────────────────────────────
#   전역 싱글톤 설정
# ─────────────────────────────────────

@dataclass(frozen=True)
class _Config:

    run                     : _RunConfig    = field(default_factory=_RunConfig)
    sys                     : _SystemConfig = field(default_factory=_SystemConfig)
    path                    : _PathConfig   = field(default_factory=_PathConfig)
    data                    : _DataConfig   = field(default_factory=_DataConfig)
    market                  : _MarketConfig = field(default_factory=_MarketConfig)
    kis                     : _KisApiConfig = field(default_factory=_KisApiConfig)
    train                   : _TrainConfig  = field(default_factory=_TrainConfig)
    model                   : _ModelConfig  = field(default_factory=_ModelConfig)
    trade                   : _TradeConfig  = field(default_factory=_TradeConfig)
    
    key                     : _KeyValues    = field(default_factory=_KeyValues)
    str                     : _StringValue  = field(default_factory=_StringValue)


config = _Config()
config.path.ensure_dirs()   # 시스템에서 필요한 디렉토리 명시적 초기화(생성)