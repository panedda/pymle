"""
Activation functions for neural network layers.

Each activation exposes:
  - forward(x)  : applies the function element-wise
  - backward(x) : derivative of the function w.r.t. its input *before* activation
                  (i.e. dA/dZ given Z, not A)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Abstract base class for activation functions."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the activation output.

        Parameters
        ----------
        x:
            Pre-activation values (Z), shape (batch, units).

        Returns
        -------
        np.ndarray
            Activated values (A), same shape as *x*.
        """

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute the element-wise derivative dA/dZ.

        Parameters
        ----------
        x:
            Pre-activation values (Z), shape (batch, units).

        Returns
        -------
        np.ndarray
            Gradient, same shape as *x*.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class Linear(Activation):
    """Identity / linear activation — dA/dZ = 1 everywhere."""

    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.ones_like(x)


class Sigmoid(Activation):
    """Logistic sigmoid: σ(x) = 1 / (1 + exp(−x))."""

    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def backward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        s = self.forward(x)
        return s * (1.0 - s)


class ReLU(Activation):
    """Rectified Linear Unit: max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.maximum(0.0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    """Leaky ReLU: max(α·x, x) where α is a small positive slope.

    Parameters
    ----------
    alpha:
        Slope for negative inputs. Default 0.01.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.where(x >= 0, x, self.alpha * x)

    def backward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.where(x >= 0, 1.0, self.alpha)

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return 1.0 - np.tanh(x) ** 2


class Softmax(Activation):
    """Softmax activation for multi-class output layers.

    Notes
    -----
    The backward method returns ones because the Softmax derivative is absorbed
    into the ``CategoricalCrossEntropy`` loss for numerical stability (the
    combined gradient simplifies to ``y_pred − y_true``). When used with any
    other loss, attach a ``Linear`` output activation instead.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        # Subtract max for numerical stability (per sample)
        shifted = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        # Jacobian is handled jointly with CCE loss; return 1 as pass-through
        return np.ones_like(x)
