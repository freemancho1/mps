from __future__ import annotations 

from dataclasses import dataclass, field 
from pathlib import Path
from zoneinfo import ZoneInfo 
from typing import Any

from . import _constants as const 


@dataclass
class SysConfig:
    seed: int = 42 
    timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo(const.CURR_TIMEZONE))
    date_format: str = const.DATE_FORMAT

    # 마켓 정보
    market_open_time: str = const.MARKET_OPEN_TIME
    market_clost_time: str = const.MARKET_CLOSE_TIME
    minutes_per_day: int = const.MINUTES_PER_DAY
    
    # 룩백 윈도우: 신호 생성 전 반드시 확보해야 할 과거 봉 수
    lookback_minutes: int = const.LOOKBACK_MINUTES
    force_close_minutes_before: int = const.FORCE_CLOSE_MINUTES_BEFORE


@dataclass
class RunBacktestConfig:
    tickers: list[str] = field(default_factory=lambda: const.TEST_TICKERS)

    start_date: str = const.TEST_START_DATE
    end_date: str = const.TEST_END_DATE    
    capital: float = const.TEST_CAPITAL

    @dataclass(frozen=True)
    class _key:
        ticker: str = "--ticker"
        start_date: str = "--start"
        end_date: str = "--end"
        capital: str = "--capital"

    @dataclass(frozen=True)
    class _msg:
        title: str = "MPS Phase 1 백테스트"
        ticker: str = "종목 코드 (기본값: 삼성전자)"
        start_date: str = "시작일 YYYY-MM-DD"
        end_date: str = "종료일 YYYY-MM-DD"
        capital: str = "초기 자본"

        @staticmethod
        def summary(args: dict[str, Any]) -> str:
            return (
                f"\n{'='*60}\n"
                f"  {args['title']}\n"
                f"    - 종목: {args['ticker']}\n"
                f"    - 기간: {args['start']} ~ {args['end']}\n"
                f"    - 초기자본: {args['capital']:,.0f}원\n"
                f"    - 보수적 왕복 비용: {args['roundtrip_cost']:.2%}\n"
                f"{'='*60}\n"
            )

    key: _key = field(default_factory=_key)
    msg: _msg = field(default_factory=_msg)


@dataclass
class LogConfig:
    base_dir: Path = field(repr=False)  # 외부 주입

    @property
    def dir(self) -> Path:
        d = self.base_dir / const.LOG_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass(frozen=True)
class CostConfig:
    """ 
    보수적 거래 비용 모델.

    실제 수익에 비용을 과소평가하면 백테스트가 낙관적이 되기 때문에,
    슬리피지 0.1%는 KRX 시장가 주문 기준 현실적 상한선임.

    왕복 비용(roundtrip_cost ≒ 0.41%) → 진입 신호의 최소 기대수준
    combined_score >= 0.55 임계값도 이 비용을 커버할 수 있는 신뢰도에서 도출됨
    """
    # 0.015% 편도 (증권사별로 다름)
    commission_rate: float = const.COMMISSION_RATE    
    # 매도 시 증권거래세 
    tax_rate: float = const.TAX_RATE            
     # 슬리피지 0.1% (보수적 추정)
    slippage_rate: float = const.SLIPPAGE_RATE     

    @property 
    def roundtrip_cost(self) -> float:
        """ 왕복 거래에 소요되는 총 비용 계산을 위한 수수료율 """
        return self.commission_rate * 2 + self.tax_rate + self.slippage_rate * 2


@dataclass 
class _Config:
    # root_dir: ~/projects/mps
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    sys: SysConfig = field(default_factory=SysConfig)
    run: RunBacktestConfig = field(default_factory=RunBacktestConfig)
    log: LogConfig = field(init=False)
    cost: CostConfig = field(default_factory=CostConfig)

    def __post_init__(self):
        self.data_dir = self.root_dir / const.DATA_DIR
        self.log = LogConfig(base_dir=self.data_dir)


# ── 전역 싱글톤 ─────────────────────────
config = _Config()