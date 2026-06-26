""" 
ThresholdModel 단위 테스트

[검증 항목]
- BUY 조건: RSI < 35 + MACD 골든크로스 (prev_diff < 0, curr_diff >= 0)
- HOLD 조건: RSI 초과, 크로스오버 미발생, 직전봉 diff 양수
- confidence 공식: min(1.0, (rsi_oversold - rsi) / rsi_oversold + closeover_base)
- confidence 상한 클리핑 (rsi=0일 때 1.0 초과 방지)
- feature_contrib 키 포함 여부
- 윈도우 1행일 때 prev == last 폴백
- run() → NumericSignal 타입·필드 검증
"""
from __future__ import annotations 

import numpy as np 
import pytest 
from datetime import datetime, timezone 

from mps.core.config import cfg 
from mps.core.libs import logger 
from mps.core.types import NumericInput, NumericSignal
from mps.model.numeric.threshold import ThresholdModel


# ── 전역변수 선언 ──────────────────────

IDX = cfg.modeling.feature_idx
TICKER = "005930"
TIMESTAMP = datetime(2025, 1, 2, 9, 30, tzinfo=cfg.sys.timezone)
N_FEATURES = cfg.modeling.feature_count

# ── 헬퍼 함수 ────────────────────────

def make_input(
    rsi: float,
    macd_diff_now: float,
    macd_diff_prev: float,
    n_rows: int = 3,
) -> NumericInput:
    """ 지정한 RSI·MACD diff값을 가진 NumericInput 생성 """
    raw = np.zeros((n_rows, N_FEATURES), dtype=np.float32)
    raw[:, IDX[cfg.key.rsi_14]] = rsi 
    raw[-1, IDX[cfg.key.macd_diff]] = macd_diff_now
    if n_rows >= 2:
        raw[-2, IDX[cfg.key.macd_diff]] = macd_diff_prev

    return NumericInput(
        ticker=TICKER,
        timestamp=TIMESTAMP,
        window=raw.copy(),
        raw_window=raw,
        window_size=n_rows 
    )


class TestPredict:

    def setup_method(self):
        self.model = ThresholdModel(rsi_oversold=35.0, closeover_base=0.3)

    # ── BUY 케이스
    def test_buy_signal(self):
        # RSI 과매도 + 골든크로스 → BUY
        inp = make_input(rsi=30.0, macd_diff_now=0.1, macd_diff_prev=-0.1)
        direction, confidence, contribution = self.model.predict(inp)
        # 거래 방향 ─ BUY와 HOLD만 존재
        assert direction == cfg.str.buy
        # 신뢰도 (양수)
        assert confidence > 0.0
        logger.test(f"confidence = {confidence}")

        # confidence = min(1.0, (35 - rsi) / 35 + 0.3)
        rsi = 28.0
        expected = min(1.0, (35.0 - rsi) / 35.0 + 0.3)
        inp = make_input(rsi=rsi, macd_diff_now=0.05, macd_diff_prev=-0.05)
        direction, confidence, contribution = self.model.predict(inp)
        assert confidence == pytest.approx(round(expected, 4), abs=1e-4)
        logger.test(f"confidence2 = {confidence}")

        # rsi=0이면 공식 결과가 1.0을 초과하므로 1.0으로 클리핑돼야 함.
        inp = make_input(rsi=0.0, macd_diff_now=1.0, macd_diff_prev=-1.0)
        direction, confidence, contribution = self.model.predict(inp)
        assert confidence == 1.0

        # curr_diff >= 0까지 골든크로스(==0 도 충족)
        inp = make_input(rsi=30.0, macd_diff_now=0.0, macd_diff_prev=-1.0)
        direction, confidence, contribution = self.model.predict(inp)
        assert direction == cfg.str.buy
        
    def test_hold_signal(self):
        # rsi == 35는 조건 미충족 (< 35만 허용) → HOLD
        inp = make_input(rsi=35.0, macd_diff_now=1.0, macd_diff_prev=-1.0)
        direction, confidence, contribution = self.model.predict(inp)
        assert direction == cfg.str.hold 
        assert confidence < cfg.sys.zero 

        # rsi >= 35는 조건 미충족 (< 35만 허용) → HOLD
        inp = make_input(rsi=50.0, macd_diff_now=1.0, macd_diff_prev=-1.0)
        direction, confidence, contribution = self.model.predict(inp)
        assert direction == cfg.str.hold 
        assert confidence < cfg.sys.zero 

        # rsi < 35는 충족, 골든크로스 아님
        inp = make_input(rsi=50.0, macd_diff_now=0.1, macd_diff_prev=0.2)
        direction, confidence, contribution = self.model.predict(inp)
        assert direction == cfg.str.hold 
        assert confidence < cfg.sys.zero 
        # 데드크로스
        inp = make_input(rsi=50.0, macd_diff_now=-0.1, macd_diff_prev=0.2)
        direction, confidence, contribution = self.model.predict(inp)
        assert direction == cfg.str.hold 
        assert confidence < cfg.sys.zero 
        # 윈도우가 1개면 macd_diff_prev == macd_diff_now 이므로 골든크로스가 없어 HOLD
        inp = make_input(rsi=20.0, macd_diff_now=0.1, macd_diff_prev=-0.2, n_rows=1)
        direction, confidence, contribution = self.model.predict(inp)
        assert direction == cfg.str.hold 

    def test_feature_contribution(self):
        # BUY 인 경우
        inp = make_input(rsi=30.0, macd_diff_now=0.1, macd_diff_prev=-0.2)
        direction, confidence, contribution = self.model.predict(inp)
        for key in (cfg.key.rsi_14, cfg.key.macd_diff, cfg.key.bb_pband, cfg.key.ret_1):
            assert key in contribution 

        # HOLD 인 경우 → cfg.signal.numeric.no_signal 리턴 
        inp = make_input(rsi=50.0, macd_diff_now=0.1, macd_diff_prev=-0.1)
        direction, confidence, contribution = self.model.predict(inp)
        assert contribution == {}

        rsi = 28.5
        inp = make_input(rsi=rsi, macd_diff_now=0.1, macd_diff_prev=-0.1)
        direction, confidence, contribution = self.model.predict(inp)
        assert contribution[cfg.key.rsi_14] == pytest.approx(rsi, abs=1e-4)
        logger.test(f"confidence = {confidence}")
        logger.test(f"contribution = {contribution}")


class TestRun:

    def setup_method(self):
        self.model = ThresholdModel()

    def test_type_and_value(self):
        inp = make_input(rsi=50.0, macd_diff_now=0.0, macd_diff_prev=0.0)
        result = self.model.run(inp)
        assert isinstance(result, NumericSignal)
        logger.test(f"result type: {type(result)}")

        assert result.ticker == TICKER 
        logger.test(f"result ticker: {result.ticker} = {TICKER}")

        assert result.timestamp == TIMESTAMP
        logger.test(f"result timestamp: {result.timestamp} = {TIMESTAMP}")

        assert result.latency_ms > 0
        logger.test(f"result latency_ms: {result.latency_ms}")

        assert result.direction == cfg.str.hold 
        logger.test(f"result direction: {result.direction}")
        