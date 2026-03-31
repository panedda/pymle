"""
Weight-update algorithms (optimizers) for neural network training.

Each optimizer exposes:
  - update(params, grads) : applies the update rule in-place, returning updated params

The ``params`` and ``grads`` arguments are dicts with keys ``"W"`` and ``"b"``.
Optimizers maintain their own internal state (e.g. momentum buffers) keyed by
an integer layer index supplied by the MLP.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


Params = Dict[str, np.ndarray]  # {"W": ..., "b": ...}


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def update(self, layer_idx: int, params: Params, grads: Params) -> Params:
        """Apply one gradient-descent step.

        Parameters
        ----------
        layer_idx:
            Zero-based index of the layer being updated (used to key
            internal optimizer state per layer).
        params:
            Dict with current ``"W"`` and ``"b"`` arrays.
        grads:
            Dict with ``"W"`` and ``"b"`` gradient arrays.

        Returns
        -------
        Params
            Updated ``params`` dict (modified **in-place** and returned).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Concrete optimizers
# ---------------------------------------------------------------------------


class SGD(Optimizer):
    """Vanilla stochastic gradient descent.

    Parameters
    ----------
    lr:
        Learning rate. Default 0.01.
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def update(self, layer_idx: int, params: Params, grads: Params) -> Params:  # noqa: D102
        params["W"] -= self.lr * grads["W"]
        params["b"] -= self.lr * grads["b"]
        return params

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr})"


class SGDMomentum(Optimizer):
    """SGD with momentum.

    Parameters
    ----------
    lr:
        Learning rate. Default 0.01.
    momentum:
        Momentum coefficient. Default 0.9.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self._v: dict[int, Params] = {}

    def update(self, layer_idx: int, params: Params, grads: Params) -> Params:  # noqa: D102
        if layer_idx not in self._v:
            self._v[layer_idx] = {
                "W": np.zeros_like(params["W"]),
                "b": np.zeros_like(params["b"]),
            }
        v = self._v[layer_idx]
        v["W"] = self.momentum * v["W"] - self.lr * grads["W"]
        v["b"] = self.momentum * v["b"] - self.lr * grads["b"]
        params["W"] += v["W"]
        params["b"] += v["b"]
        return params

    def __repr__(self) -> str:
        return f"SGDMomentum(lr={self.lr}, momentum={self.momentum})"


class RMSProp(Optimizer):
    """Root Mean Square Propagation.

    Parameters
    ----------
    lr:
        Learning rate. Default 0.001.
    rho:
        Decay rate for the moving average of squared gradients. Default 0.9.
    epsilon:
        Small constant for numerical stability. Default 1e-8.
    """

    def __init__(self, lr: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8) -> None:
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self._cache: dict[int, Params] = {}

    def update(self, layer_idx: int, params: Params, grads: Params) -> Params:  # noqa: D102
        if layer_idx not in self._cache:
            self._cache[layer_idx] = {
                "W": np.zeros_like(params["W"]),
                "b": np.zeros_like(params["b"]),
            }
        c = self._cache[layer_idx]
        c["W"] = self.rho * c["W"] + (1.0 - self.rho) * grads["W"] ** 2
        c["b"] = self.rho * c["b"] + (1.0 - self.rho) * grads["b"] ** 2
        params["W"] -= self.lr * grads["W"] / (np.sqrt(c["W"]) + self.epsilon)
        params["b"] -= self.lr * grads["b"] / (np.sqrt(c["b"]) + self.epsilon)
        return params

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.lr}, rho={self.rho})"


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam).

    Parameters
    ----------
    lr:
        Learning rate. Default 0.001.
    beta1:
        Exponential decay rate for the first moment estimate. Default 0.9.
    beta2:
        Exponential decay rate for the second moment estimate. Default 0.999.
    epsilon:
        Small constant for numerical stability. Default 1e-8.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: dict[int, Params] = {}
        self._v: dict[int, Params] = {}
        self._t: dict[int, int] = {}

    def update(self, layer_idx: int, params: Params, grads: Params) -> Params:  # noqa: D102
        if layer_idx not in self._m:
            self._m[layer_idx] = {
                "W": np.zeros_like(params["W"]),
                "b": np.zeros_like(params["b"]),
            }
            self._v[layer_idx] = {
                "W": np.zeros_like(params["W"]),
                "b": np.zeros_like(params["b"]),
            }
            self._t[layer_idx] = 0

        self._t[layer_idx] += 1
        t = self._t[layer_idx]
        m, v = self._m[layer_idx], self._v[layer_idx]

        for key in ("W", "b"):
            m[key] = self.beta1 * m[key] + (1.0 - self.beta1) * grads[key]
            v[key] = self.beta2 * v[key] + (1.0 - self.beta2) * grads[key] ** 2
            m_hat = m[key] / (1.0 - self.beta1**t)
            v_hat = v[key] / (1.0 - self.beta2**t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"
