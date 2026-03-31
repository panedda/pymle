"""
Multilayer Perceptron (MLP) — a fully-connected feedforward neural network.

Example
-------
>>> from pymle.nn.activations import ReLU, Sigmoid
>>> from pymle.nn.losses import BinaryCrossEntropy
>>> from pymle.nn.optimizers import Adam
>>> from pymle.models.mlp import MLP
>>>
>>> model = MLP(
...     layer_sizes=[2, 8, 1],
...     activations=[ReLU(), Sigmoid()],
...     loss=BinaryCrossEntropy(),
...     optimizer=Adam(lr=0.01),
... )
>>> history = model.fit(X_train, y_train, epochs=100, batch_size=32)
>>> y_pred = model.predict(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from pymle.nn.activations import Activation
from pymle.nn.losses import Loss
from pymle.nn.optimizers import Optimizer


@dataclass
class _DenseLayer:
    """Internal representation of a single fully-connected layer.

    Parameters
    ----------
    in_features:
        Number of input units.
    out_features:
        Number of output units.
    activation:
        Activation function to apply after the linear transform.
    """

    in_features: int
    out_features: int
    activation: Activation
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    # Cache populated during forward pass (needed for backprop)
    _z: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _a_in: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # He initialisation for ReLU-family, Xavier otherwise
        scale = np.sqrt(2.0 / self.in_features)
        self.W = np.random.randn(self.in_features, self.out_features) * scale
        self.b = np.zeros((1, self.out_features))

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Return a view of the layer parameters."""
        return {"W": self.W, "b": self.b}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Apply updated parameters (in-place, same objects)."""
        self.W = params["W"]
        self.b = params["b"]


class MLP:
    """Multilayer Perceptron with configurable activations, loss, and optimizer.

    Parameters
    ----------
    layer_sizes:
        Sequence of integers specifying the number of units in each layer
        **including** the input layer. For example, ``[784, 128, 64, 10]``
        creates a network with input dimension 784, two hidden layers (128
        and 64 units), and an output layer of 10 units.
    activations:
        One ``Activation`` instance per **non-input** layer. Must have
        ``len(activations) == len(layer_sizes) - 1``.
    loss:
        Loss function instance used for training.
    optimizer:
        Optimizer instance that applies the weight-update rule.
    random_state:
        Seed for NumPy's random number generator (for reproducibility).
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activations: Sequence[Activation],
        loss: Loss,
        optimizer: Optimizer,
        random_state: Optional[int] = None,
    ) -> None:
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Expected {len(layer_sizes) - 1} activations for "
                f"{len(layer_sizes)} layer sizes, got {len(activations)}."
            )
        if random_state is not None:
            np.random.seed(random_state)

        self.loss = loss
        self.optimizer = optimizer
        self._layers: List[_DenseLayer] = [
            _DenseLayer(in_f, out_f, act)
            for in_f, out_f, act in zip(layer_sizes[:-1], layer_sizes[1:], activations)
        ]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass and cache intermediate values.

        Parameters
        ----------
        X:
            Input matrix, shape (batch, input_features).

        Returns
        -------
        np.ndarray
            Output of the final layer, shape (batch, output_units).
        """
        a = X
        for layer in self._layers:
            layer._a_in = a                     # cache input to this layer
            layer._z = a @ layer.W + layer.b    # pre-activation
            a = layer.activation.forward(layer._z)
        return a

    def _backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Run backpropagation and apply optimizer updates.

        Parameters
        ----------
        y_pred:
            Output from the last forward pass, shape (batch, outputs).
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        """
        # Gradient of loss w.r.t. final activation output
        delta = self.loss.backward(y_pred, y_true)

        for idx, layer in reversed(list(enumerate(self._layers))):
            # δ * dA/dZ → gradient flowing into this layer's linear part
            dz = delta * layer.activation.backward(layer._z)

            batch_size = layer._a_in.shape[0]
            grad_W = layer._a_in.T @ dz          # shape (in, out)
            grad_b = dz.sum(axis=0, keepdims=True)  # shape (1, out)

            # Propagate delta to previous layer
            delta = dz @ layer.W.T

            # Update weights via optimizer
            params = layer.params
            updated = self.optimizer.update(idx, params, {"W": grad_W, "b": grad_b})
            layer.set_params(updated)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute network output for the given inputs (no gradient tracking).

        Parameters
        ----------
        X:
            Input matrix, shape (n_samples, input_features).

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples, output_units).
        """
        return self._forward(np.asarray(X, dtype=float))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        log_every: int = 10,
    ) -> List[float]:
        """Train the network using mini-batch gradient descent.

        Parameters
        ----------
        X:
            Training input, shape (n_samples, input_features).
        y:
            Training targets, shape (n_samples, output_units) — must be
            2-D (add a trailing axis for scalar outputs).
        epochs:
            Number of full passes over the training data.
        batch_size:
            Number of samples per mini-batch. Use ``-1`` for full-batch GD.
        verbose:
            Whether to print loss during training.
        log_every:
            Print interval (in epochs) when *verbose* is True.

        Returns
        -------
        list[float]
            Loss value recorded at the end of each epoch.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        effective_batch = n_samples if batch_size == -1 else batch_size
        history: List[float] = []

        for epoch in range(1, epochs + 1):
            # Shuffle each epoch
            perm = np.random.permutation(n_samples)
            X_s, y_s = X[perm], y[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, effective_batch):
                X_batch = X_s[start : start + effective_batch]
                y_batch = y_s[start : start + effective_batch]

                y_pred = self._forward(X_batch)
                epoch_loss += self.loss.forward(y_pred, y_batch)
                self._backward(y_pred, y_batch)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)

            if verbose and (epoch % log_every == 0 or epoch == 1):
                print(f"Epoch {epoch:>{len(str(epochs))}}/{epochs}  loss={avg_loss:.6f}")

        return history

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        layer_info = " → ".join(
            f"{l.in_features}→{l.out_features}({l.activation})"
            for l in self._layers
        )
        return (
            f"MLP({layer_info}  |  loss={self.loss}  |  optimizer={self.optimizer})"
        )
