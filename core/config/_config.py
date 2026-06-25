from __future__ import annotations 

import os 
import math 
from pathlib import Path 
from datetime import time 
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field, asdict 

from ._key_strings import _KeyValues, _StringValue
from mps.core.types import AggregationMode, SignalDirection, PatternName, TrackType


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
    # 기본 위치: ~/projects/mps ─ 이 하위에 모든 directory가 생성됨.
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
    
    @property 
    def monitoring(self) -> Path:
        # 모니터링 로그
        # output logger(signal, order) 디렉토리(logs)와 구분해 저장
        return self._artifacts_dir / "monitoring"

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
        for d in (self.monitoring, self.store, self.models):
            d.mkdir(parents=True, exist_ok=True)


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

    def to_dict(self) -> dict:
        return asdict(self)
    

@dataclass(frozen=True)
class _ModelingConfig:
    torch_device            : str           = "cuda"
    numeric_track           : TrackType     = "numeric"
    pattern_track           : TrackType     = "pattern"
    default_track           : TrackType     = numeric_track

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
    
    params                  : _HyperparameterConfig = field(default_factory=_HyperparameterConfig)
    
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
    라벨러(학습)와 ExitPolicy(청산)가 동일한 값을 공유해야,
    '모델이 학습한 것'과 '시스템이 실행하는 것'이 일치함
    
    - 익절(+2%)과 손절(-1%)이 비대칭인 이유:
      손실을 이익보다 빨리 차단해 덜 손해보기 위함.
    """
    take_profit             : float         = 0.02      # +2%
    stop_loss               : float         = 0.01      # -1%
    time_horizon            : int           = 60        # 최대 기다림 시간(60분)


@dataclass(frozen=True)
class _CostConfig:
    commission_rate         : float         = 0.00015   # 위탁 수수료 (편도 0.015%)
    tax_rate                : float         = 0.0018    # 증권 거래세(매도 시만 0.18%)
    slippage_rate           : float         = 0.001     # 슬리피지 (편도 0.1%)
    
    @property 
    def roundtrip_rate(self) -> float:
        """ 왕복 비용 비율 (수수료*2 + 세금 + 슬리피지*2) = 0.41% """
        return (self.commission_rate + self.slippage_rate) * 2 + self.tax_rate


@dataclass(frozen=True)
class _NumericSignalConfig:
    # 수치 트랙 전체 가중치
    weight                  : float         = 0.5
    
    # 무신호 기본값(field로 처리)
    no_signal               : tuple[SignalDirection, float, dict] = field(
                                default_factory=lambda: ("HOLD", 0.0, {})
                            )
    
    # Phase-1 롤 모델 용
    rsi_oversold            : float         = 35.0      # RSI 과매도 경계(이하면 BUY 후보)
    rsi_overbought          : float         = 65.0      # RSI 과매수 경계(롱 온리에선 미사용)
    rsi_closeover_base      : float         = 0.3       # 크로스오버 베이스 신뢰도
    
    
@dataclass(frozen=True)
class _PatternSignalConfig:
    # 패턴 트랙 전체 가중치
    weight                  : float         = 0.5
    
    # 무신호 기본값
    no_signal               : tuple[SignalDirection, float, PatternName] = (
                                "HOLD", 0.0, "NONE"
                            )
    
    bar1_confidence         : float         = 0.6       # 단봉 (가장 낮음)
    bar2_confidence         : float         = 0.7       # 이중봉
    bar3_confidence         : float         = 0.75      # 삼봉 (가장 높음)
    
    chart_confidence        : float         = 0.65      # 21개 봉으로 조사(잘 안나옴)
    chart_min_size          : int           = 21
    

@dataclass(frozen=True)
class _SignalConfig:
    numeric                 : _NumericSignalConfig = field(default_factory=_NumericSignalConfig)
    pattern                 : _PatternSignalConfig = field(default_factory=_PatternSignalConfig)
    
    # [수익률 레버] 결합 정책 ─ v3.1 변경
    #   - "CONFLUENCE"(구 require_confluence=True): 두 트랙 모두 BUY일 때만 진입(AND).
    #       두 약분류기의 AND는 거래가 극단적으로 고갈되고, 한 축만 강하게 맞은 기회를 버림.
    #   - "WEIGHTED"(기본): 한쪽만 BUY여도 진입 허용(soft-OR).
    #       두 트랙 동시 BUY 시 confluence_bonus를 가산해 '교차검증 합의'를 점수로 보상.
    #       단일 트랙 진입도 ScoreFilter(min_combined_score=0.55)를 넘어야 하므로
    #       '약한 신호 남발'은 점수 게이트가 계속 차단함.
    aggregation_mode        : AggregationMode = "WEIGHTED"
    # 두 트랙이 동시에 BUY일 때 combined_score에 더하는 합의 보너스(상한 1.0로 클립)
    #   - "수치·패턴이 서로 검증한다.(Cross-Verification)" 원칙을 하드 게이트가 아닌
    #     "점수 가산" 방식으로 인코딩 → 합의 신호가 단일 신호보다 우선 통과됨.
    confluence_bonus        : float         = 0.1
    
    # 최대 지연 시간 (실제 KIS API를 이용할 경우 테스트를 통합 조정 필요)
    max_latency_ms          : float         = 5_000.0
    
    
@dataclass(frozen=True)
class _RiskConfig:
    _barrier                : _BarrierConfig = field(default_factory=_BarrierConfig)
    _cost                   : _CostConfig   = field(default_factory=_CostConfig)
    
    # 1회 투입가능 금액 상한선 = 초기 자본의 10%
    max_capital_pct         : float         = 0.1
    
    # [수익성-B] 진입 컷오프: 강제청산(15:15) N분 전부터 신규 진입 금지.
    # - 마감 임박 진입은 보유 가능 시간이 짧아 익절 확률은 낮고,
    #   왕복 비용은 확정(=0.41%)으로 구조적으로 손실 거래일 가능성이 높음
    entry_cutoff_minutes    : int           = 30
    
    # [수익성-C] 손익분기 신뢰도 위에 얹는 안전 마진
    # - 최종 임계값 min_combined_score에서 
    #   (sl+왕복비용)/(tp+sl)+score_margin으로 산출 = 0.550
    score_margin            : float         = 0.08
    
    # [수익성-D] 신뢰도에 비례한 베팅 사이즈 조정
    # - Kelly 이전의 보수적 확신 비례 베팅에 사용(Phase-3+)
    use_conviction_sizing   : bool          = True
    conviction_min_factor   : float         = 0.7
    conviction_max_factor   : float         = 1.3
    
    # [수익성-E] 브레이크이븐 스톱: 미실현 + Trigger 도달 시 "다읍 봉부터"
    # - 손절선을 '진입가 * (1 + buffer)'로 상향
    # - buffer(+0.5%) > 왕복비용(0.41%) → 되밀려도 본전 이상으로 마감
    use_breakeven_stop      : float         = True 
    breakeven_trigger       : float         = 0.01      # +1.0% 도달 시
    breakeven_buffer        : float         = 0.005     # 스톱을 진입가 +0.5%로 지정
    
    @property 
    def breakeven_confidence(self) -> float:
        """ 
        [수익성-C] 손익분기 신뢰도 ─ 기대값이 0이 되면 BUY 확률 p*
        
        TakeProfit/StopLoss 두 결과만 가정한 단순 기대값 모델:
          - EV = p·TP - (1-p)·SL - 왕복비용 = 0
            → p* = (SL + 비용) / (TP + SL)
          - 기본 파라미터(2% / 1% / 0.41%)에서 p* = 0.470
        """
        return (self._barrier.stop_loss + self._cost.roundtrip_rate) \
             / (self._barrier.take_profit + self._barrier.stop_loss)
             
    # 신호통과 임계값 = 손익분기 신뢰도 + 안전마진 = 0.47 + 0.08 = 0.55
    @property 
    def min_combined_score(self) -> float:
        return self.breakeven_confidence * self.score_margin             
    
    # [수익성-F] 일일 손실 한도: 당일 실현 손실이 자본의 -5%를 넘으면
    #            당일 신규 진입 중단 (레짐 급변 시 연손 손절 차단)
    daily_loss_limit_pct    : float         = 0.05      # 5%
    

# ─────────────────────────────────────
#   데이터 상수 ─ 룩백·합성 데이터 등 데이터 관련 상수
# ─────────────────────────────────────
@dataclass(frozen=True)
class _DataConfig:
    # 캐시 데이터를 이용하지 않고 강제로 다시 데이터를 읽어오는 옵션
    force_refresh_data      : bool          = False 
    
    # 룩백: 입력 윈도우의 크기(얼마나 과거까지 입력으로 넣을것인가?) (120~240분 봉)
    # 120은 하루 거래량 390의 1/3이고 2시간(120분) 봉
    lookback_minutes        : int           = 120
    # 워밍업 봉 수: 시뮬레이터 버퍼(=lookback_minutes + warmup_bars) 계산을 위한 버퍼 공간
    warmup_bars             : int           = 50
    
    # pykrx 일봉 생성 시 옵션
    volume_weight_09        : float         = 2.6   # 개장 30분 거래량 비율
    volume_weight_12        : float         = 1.5   # 점심시간 거래량 비율
    volume_weight_15        : float         = 2.3   # 패장 30분 거래량 비율
    min_max_noise           : float         = 0.5   # 저·고가 노이즈 비율
    
    # Direction To Index or Index To Direction
    dir2idx                 : dict[SignalDirection, int] = field(
                                default_factory=lambda: {"BUY": 0, "HOLD": 1}
                            )
    idx2dir                 : dict[int, SignalDirection] = field(
                                default_factory=lambda: {0: "BUY", 1: "HOLD"}
                            )
    
    # ── 교체 파생값 정의 ─────────────────────────
    _market                 : _MarketConfig = field(default_factory=_MarketConfig)
    _barrier                : _BarrierConfig= field(default_factory=_BarrierConfig)
    
    @property 
    def buffer_days(self) -> int:
        """ 룩백 충족에 필요한 워밍업 '거래일' 수 (Walk-Forward 버퍼) """
        return math.ceil(self.lookback_minutes / self._market.minutes_per_day) + 1 # = 2
    
    @property 
    def buffer_bars(self) -> int:
        """ 시뮬레이터 deque 버퍼 크기 = 룩백 + 지표 워밍업 봉 수 """
        return self.lookback_minutes + self.warmup_bars
    
    @property 
    def embargo_bars(self) -> int:
        """ 
        train/val 경계 엠바고 (봉 수) = lookback + horizon = 120 + 60 = 180
        
        train 마지막 샘플의 '라벨'은 horizon분 뒤 봉까지,
        val 첫 샘플의 '윈도우'는 lookback분 앞 봉까지 참조하므로,
        이 부분은 비워야 같은 봉을 train과 val에서 공유하는 누수가 사라짐.
        """
        return self.lookback_minutes + self._barrier.time_horizon
    
    
# ─────────────────────────────────────
#   전역 싱글톤 설정
# ─────────────────────────────────────

@dataclass(frozen=True)
class _Config:

    run         : _RunConfig            = field(default_factory=_RunConfig)
    sys         : _SystemConfig         = field(default_factory=_SystemConfig)
    path        : _PathConfig           = field(default_factory=_PathConfig)
    
    kis         : _KisApiConfig         = field(default_factory=_KisApiConfig)
        
    market      : _MarketConfig         = field(default_factory=_MarketConfig)
    
    lstm        : _LSTMConfig           = field(default_factory=_LSTMConfig)
    cnn         : _CNNConfig            = field(default_factory=_CNNConfig)
    params      : _HyperparameterConfig = field(default_factory=_HyperparameterConfig)
    modeling    : _ModelingConfig       = field(default_factory=_ModelingConfig)
    
    barrier     : _BarrierConfig        = field(default_factory=_BarrierConfig)
    cost        : _CostConfig           = field(default_factory=_CostConfig)
    signal      : _SignalConfig         = field(default_factory=_SignalConfig)
    risk        : _RiskConfig           = field(default_factory=_RiskConfig)
    
    data        : _DataConfig           = field(default_factory=_DataConfig)
    
    key         : _KeyValues            = field(default_factory=_KeyValues)
    str         : _StringValue          = field(default_factory=_StringValue)


config = _Config()
config.path.ensure_dirs()   # 시스템에서 필요한 디렉토리 명시적 초기화(생성)    