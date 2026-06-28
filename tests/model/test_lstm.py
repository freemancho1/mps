from __future__ import annotations 

import tempfile
import pytest
import torch
import numpy as np
from typing import cast
from datetime import datetime
from pathlib import Path 
from torch.utils.data import Dataset 

from mps.core.config import cfg 
from mps.core.types import NumericInput, NumericSignal, TrainHistory
from mps.core.libs import logger 
from mps.model.numeric.lstm import LSTMModel, LSTMNet
from mps.model import ModelTrainer


# ── 공통 상수 정의:
DEVICE = cfg.modeling.torch_device
WIN_SIZE = 10
N_FEATURES = cfg.modeling.feature_count
TICKER = cfg.run.tickers[0]
TIMESTAMP = datetime(2025, 1, 5, 9, 30, tzinfo=cfg.sys.timezone)

# ── 헬퍼함수 정의:
def make_input(win_size: int = WIN_SIZE) -> NumericInput:
    rng = np.random.default_rng(seed=cfg.sys.seed)
    window = rng.standard_normal((win_size, N_FEATURES)).astype(np.float32)
    return NumericInput(
        ticker=TICKER,
        timestamp=TIMESTAMP,
        window=window,
        raw_window=window.copy(),
        window_size=win_size,
    )


class FakeDataset(Dataset):
    """ 학습용 최소 합성 데이터셋 (BUY 20% / HOLD 80%) """
    def __init__(self, n_samples: int = 120, win_size: int = WIN_SIZE) -> None:
        rng = np.random.default_rng(seed=cfg.sys.seed)
        X = rng.standard_normal((n_samples, win_size, N_FEATURES)).astype(np.float32)
        y = np.ones(n_samples, dtype=np.int64)      # HOLD = 1
        y[: n_samples // 5] = 0                     # BUY = 0 (처음 1/5개를 0으로, 20%)
        self._X = torch.from_numpy(X)
        self._y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self._y)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y[idx]
    
    def class_counts(self) -> np.ndarray:
        return np.bincount(self._y.numpy(), minlength=2)
    

@pytest.fixture 
def fast_trainer(monkeypatch):
    """ epochs=3, patience=2로 학습시간을 단축하는 픽스처 """
    from mps.core.config._config import _HyperparameterConfig
    original = cfg.params
    fast = _HyperparameterConfig(
        epochs=3, patience=2,
        batch_size=original.batch_size,
        lr=original.lr,
        weight_decay=original.weight_decay,
        val_ratio=original.val_ratio
    )
    object.__setattr__(cfg, "params", fast)
    yield ModelTrainer(device=DEVICE)
    object.__setattr__(cfg, "params", original)


@pytest.fixture 
def net():
    return LSTMNet(
        input_size=N_FEATURES, hidden_size=16,
        num_layers=1, num_classes=2, dropout=0.0,
    )


@pytest.fixture 
def model():
    return LSTMModel(
        device=DEVICE,
        model_net_params=dict(
            input_size=N_FEATURES, hidden_size=16,
            num_layers=1, num_classes=2, dropout=0.0,
        )
    )


# ── TEST:

class TestLSTMNet:

    def test_output_shape_batch(self, net):
        batch_size = 1
        x = torch.randn(batch_size, WIN_SIZE, N_FEATURES)
        out = net(x)
        assert out.shape == (batch_size, 2)  # (batch_size, num_classes)
        logger.test(f"BATCH_SIZE = {batch_size} → 출력결과 shape = {out.shape}")

        batch_size = 10     
        x = torch.randn(batch_size, WIN_SIZE, N_FEATURES)
        out = net(x)
        assert out.shape == (batch_size, 2)  # (batch_size, num_classes)
        logger.test(f"BATCH_SIZE = {batch_size} → 출력결과 shape = {out.shape}")
        assert out.dtype == torch.float32 

        batch_size = 20
        num_classes = 3
        net = LSTMNet(
            input_size=N_FEATURES, hidden_size=32,
            num_layers=5, num_classes=num_classes, dropout=0.0
        )
        out = net(torch.randn(batch_size, WIN_SIZE, N_FEATURES))
        assert out.shape == (batch_size, num_classes)  # (batch_size, num_classes)
        logger.test(f"BATCH_SIZE = {batch_size}, NUM_CLASSES = {num_classes} → 출력결과 shape = {out.shape}")        


class TestLSTMModel:

    def test_not_trained_model(self, model):
        assert model.is_trained is False 
        logger.test(f"학습되지 않은 모델 is_trained = {model.is_trained}")

        assert isinstance(model.model, LSTMNet)
        logger.test(f"생성된 모델 타입 = {type(model.model)}")

        assert not model.model.training 
        logger.test(f"추론 모드로 모델 생성: model.model.training = False")

    def test_weights_path(self, net):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as file:
            path = Path(file.name)
        torch.save({cfg.key.state_dict: net.state_dict()}, path)
        logger.test(f"weights path = {path}")
        logger.test(f"state_dict: = {net.state_dict().keys()}")

        try:
            model = LSTMModel(weights_path=path, device=DEVICE, model_net_params=dict(
                input_size=N_FEATURES, hidden_size=16, 
                num_layers=1, num_classes=2, dropout=0.0
            ))
            assert model.is_trained is True 
        finally:
            path.unlink(missing_ok=True)

    def _force_buy(self, model: LSTMModel):
        """ 모델 출력의 BUY(class=0) 로짓을 강제로 높여 BUY 신호로 유도. """
        with torch.no_grad():
            for p in model.model.head.parameters():
                p.zero_()
            # head 마지막 Linear의 bias: BUY 클래스(0)에 큰 값 부여
            last_linear = cast(
                torch.nn.Linear, 
                list(model.model.head.children())[-1]
            )
            last_linear.bias[0] = 10.0
            last_linear.bias[1] = -10.0

    def _force_hold(self, model: LSTMModel):
        """HOLD(class=1) 로짓을 강제로 높여 HOLD 신호를 유도."""
        with torch.no_grad():
            for p in model.model.head.parameters():
                p.zero_()
            last_linear = cast(
                torch.nn.Linear, 
                list(model.model.head.children())[-1]
            )
            last_linear.bias[0] = -10.0
            last_linear.bias[1] = 10.0            

    def test_return_tuple(self, model):
        result = model.predict(make_input())
        assert len(result) == 3


class TestLSTMModelPredict:

    def _force_buy(self, model: LSTMModel):
        """모델 출력의 BUY(cls=0) 로짓을 강제로 높여 BUY 신호를 유도."""
        with torch.no_grad():
            # weight는 유지해야 backward 시 gradient가 흐름
            last_linear = cast(
                torch.nn.Linear,
                list(model.model.head.children())[-1]
            )
            last_linear.bias[0] = 10.0
            last_linear.bias[1] = -10.0

    def _force_hold(self, model: LSTMModel):
        """HOLD(cls=1) 로짓을 강제로 높여 HOLD 신호를 유도."""
        with torch.no_grad():
            last_linear = cast(
                torch.nn.Linear,
                list(model.model.head.children())[-1]
            )
            last_linear.bias[0] = -10.0
            last_linear.bias[1] = 10.0

    def test_returns_tuple_of_3(self, model):
        result = model.predict(make_input())
        assert len(result) == 3

    def test_direction_is_valid(self, model):
        direction, _, _ = model.predict(make_input())
        assert direction in ("BUY", "HOLD")

    def test_confidence_in_range(self, model):
        _, conf, _ = model.predict(make_input())
        assert 0.0 <= conf <= 1.0

    def test_hold_contrib_is_empty(self, model):
        self._force_hold(model)
        direction, conf, contrib = model.predict(make_input())
        assert direction == "HOLD"
        assert conf == 0.0
        assert contrib == {}

    def test_buy_contrib_has_all_features(self, model):
        self._force_buy(model)
        direction, _, contrib = model.predict(make_input())
        assert direction == "BUY"
        for name in cfg.modeling.feature_names:
            assert name in contrib

    def test_buy_contrib_values_sum_to_1(self, model):
        self._force_buy(model)
        _, _, contrib = model.predict(make_input())
        assert sum(contrib.values()) == pytest.approx(1.0, abs=1e-3)

    def test_buy_no_attribution_contrib_empty(self, model):
        """attribute=False이면 BUY여도 contrib은 빈 dict."""
        self._force_buy(model)
        m = LSTMModel(device=DEVICE, attribute=False, model_net_params=dict(
            input_size=N_FEATURES, hidden_size=16,
            num_layers=1, num_classes=2, dropout=0.0,
        ))
        self._force_buy(m)
        _, _, contrib = m.predict(make_input())
        assert contrib == {}

    def test_confidence_rounded_to_4_decimals(self, model):
        _, conf, _ = model.predict(make_input())
        assert conf == round(conf, 4)


# ─── LSTMModel.run() ──────────────────────────────────────────────────────

class TestLSTMModelRun:

    def test_returns_numeric_signal(self, model):
        assert isinstance(model.run(make_input()), NumericSignal)

    def test_ticker_propagated(self, model):
        assert model.run(make_input()).ticker == TICKER

    def test_timestamp_propagated(self, model):
        assert model.run(make_input()).timestamp == TIMESTAMP

    def test_latency_ms_non_negative(self, model):
        assert model.run(make_input()).latency_ms >= 0.0

    def test_direction_valid(self, model):
        assert model.run(make_input()).direction in ("BUY", "HOLD")

    def test_feature_contrib_is_dict(self, model):
        assert isinstance(model.run(make_input()).feature_contrib, dict)


# ─── LSTMModel.build_contrib() ────────────────────────────────────────────

class TestBuildContrib:

    def test_keys_match_feature_names(self):
        saliency = np.ones(N_FEATURES, dtype=np.float32)
        contrib = LSTMModel.build_contrib(saliency)
        assert set(contrib.keys()) == set(cfg.modeling.feature_names)

    def test_values_sum_to_1(self):
        saliency = np.abs(np.random.default_rng(0).standard_normal(N_FEATURES).astype(np.float32))
        contrib = LSTMModel.build_contrib(saliency)
        assert sum(contrib.values()) == pytest.approx(1.0, abs=1e-4)

    def test_all_zero_saliency_no_division_error(self):
        """전체 기여도가 0이면 1.0으로 대체해 ZeroDivisionError 없이 동작."""
        contrib = LSTMModel.build_contrib(np.zeros(N_FEATURES, dtype=np.float32))
        assert all(v == 0.0 for v in contrib.values())

    def test_single_nonzero_feature(self):
        """하나의 피처만 기여 → 해당 피처 contrib == 1.0."""
        saliency = np.zeros(N_FEATURES, dtype=np.float32)
        saliency[0] = 1.0
        contrib = LSTMModel.build_contrib(saliency)
        assert contrib[cfg.modeling.feature_names[0]] == pytest.approx(1.0, abs=1e-4)
        assert all(v == 0.0 for k, v in contrib.items() if k != cfg.modeling.feature_names[0])

    def test_values_rounded_to_4_decimals(self):
        saliency = np.random.default_rng(1).standard_normal(N_FEATURES).astype(np.float32)
        saliency = np.abs(saliency)
        contrib = LSTMModel.build_contrib(saliency)
        for v in contrib.values():
            assert v == round(v, 4)


# ─── LSTMModel.from_net() ─────────────────────────────────────────────────

@pytest.fixture
def cfg_net():
    """ cfg.lstm 기본값과 동일한 구조의 LSTMNet (from_net 테스트용) """
    return LSTMNet(**cfg.lstm.to_dict())


class TestFromNet:

    def test_is_trained_after_from_net(self, cfg_net):
        m = LSTMModel.from_net(cfg_net, device=DEVICE)
        assert m.is_trained is True

    def test_model_in_eval_mode(self, cfg_net):
        m = LSTMModel.from_net(cfg_net, device=DEVICE)
        assert not m.model.training

    def test_weights_match_original_net(self, cfg_net):
        m = LSTMModel.from_net(cfg_net, device=DEVICE)
        for (k1, v1), (k2, v2) in zip(
            cfg_net.state_dict().items(), m.model.state_dict().items()
        ):
            assert k1 == k2
            assert torch.allclose(v1.cpu(), v2.cpu())


# ─── ModelTrainer ─────────────────────────────────────────────────────────

class TestModelTrainer:

    def test_raises_type_error_for_non_sized(self):
        trainer = ModelTrainer(device=DEVICE)
        with pytest.raises(TypeError):
            trainer.train(LSTMNet(), iter([]))     # iterator는 Sized 아님

    def test_raises_value_error_for_small_dataset(self):
        trainer = ModelTrainer(device=DEVICE)
        tiny = FakeDataset(n_samples=cfg.modeling.min_dataset_size - 1, win_size=WIN_SIZE)
        with pytest.raises(ValueError):
            trainer.train(LSTMNet(), tiny)

    def test_returns_model_and_history(self, fast_trainer):
        net = LSTMNet(input_size=N_FEATURES, hidden_size=16,
                      num_layers=1, num_classes=2, dropout=0.0)
        result = fast_trainer.train(net, FakeDataset())
        model, history = result
        assert isinstance(model, LSTMNet)
        assert isinstance(history, TrainHistory)

    def test_history_lists_not_empty(self, fast_trainer):
        net = LSTMNet(input_size=N_FEATURES, hidden_size=16,
                      num_layers=1, num_classes=2, dropout=0.0)
        _, history = fast_trainer.train(net, FakeDataset())
        assert len(history.train_loss) > 0
        assert len(history.val_loss) > 0
        assert len(history.val_acc) > 0

    def test_best_epoch_set(self, fast_trainer):
        net = LSTMNet(input_size=N_FEATURES, hidden_size=16,
                      num_layers=1, num_classes=2, dropout=0.0)
        _, history = fast_trainer.train(net, FakeDataset())
        assert history.best_epoch >= 0

    def test_model_in_eval_mode_after_train(self, fast_trainer):
        net = LSTMNet(input_size=N_FEATURES, hidden_size=16,
                      num_layers=1, num_classes=2, dropout=0.0)
        trained_net, _ = fast_trainer.train(net, FakeDataset())
        assert not trained_net.training


# ─── End-to-End: 학습 → 추론 ─────────────────────────────────────────────

class TestEndToEnd:

    def test_train_then_predict(self, fast_trainer):
        """LSTMNet 학습 → from_net() 래핑 → predict() 전 과정."""
        net = LSTMNet(**cfg.lstm.to_dict())
        trained_net, history = fast_trainer.train(net, FakeDataset())

        inference_model = LSTMModel.from_net(trained_net, device=DEVICE)
        direction, conf, contrib = inference_model.predict(make_input())

        assert direction in ("BUY", "HOLD")
        assert 0.0 <= conf <= 1.0

    def test_train_then_run(self, fast_trainer):
        """학습 후 run() 이 NumericSignal 을 올바르게 반환."""
        net = LSTMNet(**cfg.lstm.to_dict())
        trained_net, _ = fast_trainer.train(net, FakeDataset())

        inference_model = LSTMModel.from_net(trained_net, device=DEVICE)
        signal = inference_model.run(make_input())

        assert isinstance(signal, NumericSignal)
        assert signal.ticker == TICKER
        assert signal.direction in ("BUY", "HOLD")
        assert signal.latency_ms >= 0.0

    def test_save_and_load_then_predict(self, fast_trainer):
        """학습 → 가중치 저장 → LSTMModel 로드 → predict() 왕복 검증."""
        net = LSTMNet(input_size=N_FEATURES, hidden_size=16,
                      num_layers=1, num_classes=2, dropout=0.0)
        trained_net, _ = fast_trainer.train(net, FakeDataset())

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        torch.save({cfg.key.state_dict: trained_net.state_dict()}, path)

        try:
            loaded = LSTMModel(
                weights_path=path, device=DEVICE,
                model_net_params=dict(input_size=N_FEATURES, hidden_size=16,
                                num_layers=1, num_classes=2, dropout=0.0),
            )
            assert loaded.is_trained is True
            direction, conf, _ = loaded.predict(make_input())
            assert direction in ("BUY", "HOLD")
            assert 0.0 <= conf <= 1.0
        finally:
            path.unlink(missing_ok=True)

