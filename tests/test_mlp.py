"""Unit tests for the MLP and all nn building blocks."""

from __future__ import annotations

import numpy as np
import pytest

from pymle.nn.activations import LeakyReLU, Linear, ReLU, Sigmoid, Softmax, Tanh
from pymle.nn.losses import BinaryCrossEntropy, CategoricalCrossEntropy, MAE, MSE
from pymle.nn.optimizers import Adam, RMSProp, SGD, SGDMomentum
from pymle.models.mlp import MLP

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_regression_data(n: int = 64, d: int = 4) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.standard_normal((n, d)).astype(float)
    y = (X @ RNG.standard_normal(d) + 0.1 * RNG.standard_normal(n)).reshape(-1, 1)
    return X, y


def _make_binary_data(n: int = 64, d: int = 4) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.standard_normal((n, d)).astype(float)
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    return X, y


def _make_multiclass_data(
    n: int = 64, d: int = 4, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.standard_normal((n, d)).astype(float)
    labels = RNG.integers(0, k, size=n)
    y = np.zeros((n, k))
    y[np.arange(n), labels] = 1.0
    return X, y


# ---------------------------------------------------------------------------
# Activation tests
# ---------------------------------------------------------------------------

ACTIVATIONS = [Linear(), Sigmoid(), ReLU(), LeakyReLU(0.1), Tanh(), Softmax()]


@pytest.mark.parametrize("act", ACTIVATIONS, ids=lambda a: type(a).__name__)
def test_activation_forward_shape(act):
    x = RNG.standard_normal((8, 16))
    out = act.forward(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("act", ACTIVATIONS[:-1], ids=lambda a: type(a).__name__)  # skip Softmax
def test_activation_backward_shape(act):
    x = RNG.standard_normal((8, 16))
    grad = act.backward(x)
    assert grad.shape == x.shape


def test_relu_no_negative_output():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert (ReLU().forward(x) >= 0).all()


def test_sigmoid_range():
    x = RNG.standard_normal((100,))
    s = Sigmoid().forward(x)
    assert ((s > 0) & (s < 1)).all()


def test_softmax_sums_to_one():
    x = RNG.standard_normal((8, 5))
    out = Softmax().forward(x)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(8), atol=1e-6)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

def test_mse_zero_on_perfect_prediction():
    y = RNG.standard_normal((10, 1))
    assert MSE().forward(y, y) == pytest.approx(0.0)


def test_mae_zero_on_perfect_prediction():
    y = RNG.standard_normal((10, 1))
    assert MAE().forward(y, y) == pytest.approx(0.0)


def test_bce_backward_shape():
    y_pred = np.clip(RNG.standard_normal((8, 1)) * 0.1 + 0.5, 0.01, 0.99)
    y_true = (RNG.random((8, 1)) > 0.5).astype(float)
    grad = BinaryCrossEntropy().backward(y_pred, y_true)
    assert grad.shape == y_pred.shape


def test_cce_backward_shape():
    y_pred = Softmax().forward(RNG.standard_normal((8, 4)))
    y_true = np.zeros((8, 4))
    y_true[np.arange(8), RNG.integers(0, 4, 8)] = 1.0
    grad = CategoricalCrossEntropy().backward(y_pred, y_true)
    assert grad.shape == y_pred.shape


# ---------------------------------------------------------------------------
# MLP forward pass tests
# ---------------------------------------------------------------------------

def test_mlp_output_shape_regression():
    X, y = _make_regression_data()
    model = MLP([4, 8, 1], [ReLU(), Linear()], MSE(), SGD(lr=0.01), random_state=0)
    out = model.predict(X)
    assert out.shape == (64, 1)


def test_mlp_output_shape_multiclass():
    X, y = _make_multiclass_data()
    model = MLP([4, 8, 3], [ReLU(), Softmax()], CategoricalCrossEntropy(), Adam(), random_state=0)
    out = model.predict(X)
    assert out.shape == (64, 3)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(64), atol=1e-6)


# ---------------------------------------------------------------------------
# Training convergence tests
# ---------------------------------------------------------------------------

OPTIMIZERS = [SGD(lr=0.01), SGDMomentum(lr=0.01), RMSProp(lr=0.001), Adam(lr=0.001)]


@pytest.mark.parametrize("opt", OPTIMIZERS, ids=lambda o: type(o).__name__)
def test_training_reduces_loss_regression(opt):
    """Loss should decrease over 200 epochs on a simple regression task."""
    np.random.seed(0)
    X, y = _make_regression_data(n=128)
    model = MLP([4, 16, 1], [ReLU(), Linear()], MSE(), opt, random_state=0)
    history = model.fit(X, y, epochs=200, batch_size=32, verbose=False)
    assert history[-1] < history[0], (
        f"{type(opt).__name__}: loss did not decrease ({history[0]:.4f} → {history[-1]:.4f})"
    )


def test_training_reduces_loss_binary_classification():
    np.random.seed(0)
    X, y = _make_binary_data(n=128)
    model = MLP([4, 8, 1], [ReLU(), Sigmoid()], BinaryCrossEntropy(), Adam(lr=0.01), random_state=0)
    history = model.fit(X, y, epochs=200, batch_size=32, verbose=False)
    assert history[-1] < history[0]


def test_mlp_xor():
    """Classic XOR test — a network with one hidden layer must learn XOR."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    np.random.seed(7)
    model = MLP([2, 8, 1], [Tanh(), Sigmoid()], BinaryCrossEntropy(), Adam(lr=0.05), random_state=7)
    model.fit(X, y, epochs=2000, batch_size=4, verbose=False)
    preds = (model.predict(X) > 0.5).astype(int)
    assert (preds == y).all(), f"XOR not learned. Predictions: {preds.ravel()}"


# ---------------------------------------------------------------------------
# Gradient check (numerical differentiation)
# ---------------------------------------------------------------------------

def _numerical_gradient(
    model: MLP, X: np.ndarray, y: np.ndarray, eps: float = 1e-5
) -> list[dict[str, np.ndarray]]:
    """Compute numerical gradients for all parameters via finite differences."""
    num_grads = []
    for layer in model._layers:
        ng = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
        for key, arr in [("W", layer.W), ("b", layer.b)]:
            it = np.nditer(arr, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = arr[idx]
                arr[idx] = orig + eps
                loss_plus = model.loss.forward(model._forward(X), y)
                arr[idx] = orig - eps
                loss_minus = model.loss.forward(model._forward(X), y)
                arr[idx] = orig
                ng[key][idx] = (loss_plus - loss_minus) / (2 * eps)
                it.iternext()
        num_grads.append(ng)
    return num_grads


def test_gradient_check():
    """Analytical gradients should match numerical ones within tolerance."""
    np.random.seed(123)
    X = RNG.standard_normal((4, 3))
    y = RNG.standard_normal((4, 2))
    model = MLP([3, 4, 2], [Tanh(), Linear()], MSE(), SGD(lr=0.0), random_state=123)

    # One forward+backward pass to compute analytical grads
    y_pred = model._forward(X)

    # Store analytical gradients before optimizer modifies weights
    analytical = []
    delta = model.loss.backward(y_pred, y)
    for layer in reversed(model._layers):
        dz = delta * layer.activation.backward(layer._z)
        grad_W = layer._a_in.T @ dz
        grad_b = dz.sum(axis=0, keepdims=True)
        delta = dz @ layer.W.T
        analytical.insert(0, {"W": grad_W, "b": grad_b})

    numerical = _numerical_gradient(model, X, y)

    for i, (ana, num) in enumerate(zip(analytical, numerical)):
        for key in ("W", "b"):
            rel_err = np.abs(ana[key] - num[key]) / (np.abs(num[key]) + 1e-8)
            assert rel_err.max() < 1e-4, (
                f"Layer {i} {key}: max relative error = {rel_err.max():.2e}"
            )
