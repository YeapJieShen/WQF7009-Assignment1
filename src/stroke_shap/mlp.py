"""A small PyTorch multilayer perceptron with a scikit-learn interface."""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn

from stroke_shap import SEED


class MLP(nn.Module):
    """Fully connected network with ReLU hidden layers and a sigmoid output.

    Args:
        n_features: Number of input features.
        hidden: Width of each hidden layer.
        dropout: Dropout rate after the first hidden layer.
    """

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (32, 16), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for i, size in enumerate(hidden):
            layers += [nn.Linear(width, size), nn.ReLU()]
            if i == 0:
                layers.append(nn.Dropout(dropout))
            width = size
        layers += [nn.Linear(width, 1), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the stroke probability for each row of ``x``, shape (n, 1)."""
        return self.layers(x)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Train :class:`MLP` with Adam and binary cross-entropy, scikit-learn style.

    Wrapping the network as an estimator lets it share pipelines, cross-validation
    and metrics with the random forest. The fitted network is ``module_``.

    Args:
        hidden: Width of each hidden layer.
        dropout: Dropout rate after the first hidden layer.
        learning_rate: Adam learning rate.
        weight_decay: Adam L2 penalty.
        epochs: Passes over the training data.
        batch_size: Rows per gradient step.
        random_state: Seed for weight initialisation, dropout and shuffling.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (32, 16),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 128,
        random_state: int = SEED,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        """Train a new network on ``X`` (n, features) and binary labels ``y``."""
        torch.manual_seed(self.random_state)
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_t.shape[1]
        self.module_ = MLP(self.n_features_in_, self.hidden, self.dropout)
        optimiser = torch.optim.Adam(
            self.module_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        shuffle = torch.Generator().manual_seed(self.random_state)

        self.module_.train()
        for _ in range(self.epochs):
            for batch in torch.randperm(len(X_t), generator=shuffle).split(self.batch_size):
                optimiser.zero_grad()
                loss = nn.functional.binary_cross_entropy(self.module_(X_t[batch]), y_t[batch])
                loss.backward()
                optimiser.step()
        self.module_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n, 2): [no stroke, stroke]."""
        with torch.no_grad():
            stroke = self.module_(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy().ravel()
        return np.column_stack([1 - stroke, stroke])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class at the default 0.5 threshold."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
