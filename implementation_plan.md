# Multilayer Feedforward Network

A from-scratch implementation of a fully-connected neural network built purely on **NumPy** (no PyTorch/TensorFlow), designed for learning and experimentation. The design is pluggable: activation functions, loss functions, and optimizers are all swappable, independent components.

---

## Proposed File Structure

```
src/pymle/
├── models/
│   ├── __init__.py
│   └── mlp.py                  # [NEW] MLP model class
├── nn/                         # [NEW] Subpackage for NN building blocks
│   ├── __init__.py
│   ├── activations.py          # [NEW] Activation functions
│   ├── losses.py               # [NEW] Loss functions
│   └── optimizers.py           # [NEW] Weight update algorithms
tests/
└── test_mlp.py                 # [NEW] Unit tests
```

---

## Proposed Changes

### `src/pymle/nn/` (new subpackage)

#### [NEW] `activations.py`
An abstract base class `Activation` with a `forward(x)` and `backward(x)` method (all activations need both for backprop). Concrete implementations:

| Class | Description |
|---|---|
| `Sigmoid` | σ(x) = 1 / (1 + e^-x) |
| `ReLU` | max(0, x) |
| `LeakyReLU` | max(αx, x) |
| `Tanh` | tanh(x) |
| `Softmax` | e^x / Σ e^x (for output layer) |
| `Linear` | identity passthrough |

#### [NEW] `losses.py`
An abstract base class `Loss` with a `forward(y_pred, y_true)` and `backward(y_pred, y_true)` method. Concrete implementations:

| Class | Description | Typical use |
|---|---|---|
| `MSE` | Mean Squared Error | Regression |
| `MAE` | Mean Absolute Error | Regression |
| `BinaryCrossEntropy` | Binary cross-entropy | Binary classification |
| `CategoricalCrossEntropy` | Categorical cross-entropy | Multiclass classification |

#### [NEW] `optimizers.py`
An abstract base class `Optimizer` with an `update(layer, grad_W, grad_b)` method. Concrete implementations:

| Class | Description |
|---|---|
| `SGD` | Vanilla stochastic gradient descent |
| `SGDMomentum` | SGD with momentum |
| `RMSProp` | Root mean square propagation |
| `Adam` | Adaptive moment estimation |

---

### `src/pymle/models/mlp.py` [NEW]

The `MLP` class will be the main model. Key design decisions:
- **Layer-first**: A `DenseLayer` internal class holds weights (`W`), biases (`b`), and the assigned `Activation`.
- **Forward pass**: Iterate through layers, storing intermediate values (`z`, `a`) needed for backpropagation.
- **Backward pass**: Full backpropagation from loss gradient down through each layer, computing `grad_W` and `grad_b` at each step.
- **Optimizer step**: Each call to `backward` calls the optimizer's `update` for each layer.

```python
# Example usage
from pymle.nn.activations import ReLU, Softmax
from pymle.nn.losses import CategoricalCrossEntropy
from pymle.nn.optimizers import Adam
from pymle.models.mlp import MLP

model = MLP(
    layer_sizes=[784, 128, 64, 10],
    activations=[ReLU(), ReLU(), Softmax()],
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(lr=0.001),
)

model.fit(X_train, y_train, epochs=10, batch_size=32)
y_pred = model.predict(X_test)
```

---

### `tests/test_mlp.py` [NEW]
Unit tests covering:
- Forward pass shape correctness
- Backward pass (gradient check using numerical differentiation)
- Training with each optimizer reduces loss
- All activation functions and loss functions run without error

---

## Open Questions

> [!IMPORTANT]
> 1. **Numerical approach**: The implementation will use pure **NumPy**. Do you want it to also support GPU acceleration (e.g., via CuPy as a drop-in for NumPy) in the future? This doesn't change the API, just worth knowing.
> 2. **Batch support**: Should `fit()` support mini-batch gradient descent (default) as well as full-batch and online (single sample) modes?
> 3. **Regularization**: Should I add L1/L2 weight regularization as a parameter on `MLP`?

---

## Verification Plan

1. Run `pytest tests/test_mlp.py -v` to confirm all unit tests pass.
2. Run a quick smoke test: train an MLP on a small synthetic XOR dataset and confirm loss decreases.
