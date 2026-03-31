"""
Loss functions for neural network training.

Each loss exposes:
  - forward(y_pred, y_true)  : scalar loss value
  - backward(y_pred, y_true) : gradient of the loss w.r.t. y_pred
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the scalar loss.

        Parameters
        ----------
        y_pred:
            Model predictions, shape (batch, outputs).
        y_true:
            Ground-truth targets, same shape as *y_pred*.

        Returns
        -------
        float
            Mean loss over the batch.
        """

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of loss w.r.t. y_pred.

        Parameters
        ----------
        y_pred:
            Model predictions, shape (batch, outputs).
        y_true:
            Ground-truth targets, same shape as *y_pred*.

        Returns
        -------
        np.ndarray
            Gradient dL/dy_pred, same shape as *y_pred*.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Regression losses
# ---------------------------------------------------------------------------


class MSE(Loss):
    """Mean Squared Error: L = mean((y_pred - y_true)²)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:  # noqa: D102
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:  # noqa: D102
        # Divide by total number of elements to match np.mean in forward()
        return 2.0 * (y_pred - y_true) / y_pred.size


class MAE(Loss):
    """Mean Absolute Error: L = mean(|y_pred - y_true|)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:  # noqa: D102
        return float(np.mean(np.abs(y_pred - y_true)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:  # noqa: D102
        # Divide by total number of elements to match np.mean in forward()
        return np.sign(y_pred - y_true) / y_pred.size


# ---------------------------------------------------------------------------
# Classification losses
# ---------------------------------------------------------------------------


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy for single-output sigmoid networks.

    L = -mean(y·log(p) + (1−y)·log(1−p))
    """

    _eps: float = 1e-12

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:  # noqa: D102
        p = np.clip(y_pred, self._eps, 1.0 - self._eps)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:  # noqa: D102
        p = np.clip(y_pred, self._eps, 1.0 - self._eps)
        # Divide by total number of elements to match np.mean in forward()
        return (-(y_true / p) + (1.0 - y_true) / (1.0 - p)) / y_pred.size


class CategoricalCrossEntropy(Loss):
    """Categorical cross-entropy for Softmax output layers.

    L = -mean(Σ y·log(p))

    Notes
    -----
    The backward gradient is the simplified ``(y_pred - y_true) / batch``
    expression that results from combining the Softmax Jacobian with the
    CCE derivative, giving a numerically stable and efficient computation.
    *y_true* must be one-hot encoded.
    """

    _eps: float = 1e-12

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:  # noqa: D102
        p = np.clip(y_pred, self._eps, 1.0)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:  # noqa: D102
        # Combined Softmax + CCE gradient
        return (y_pred - y_true) / y_pred.shape[0]
