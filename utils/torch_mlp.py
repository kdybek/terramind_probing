import copy
import numpy as np
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupShuffleSplit


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layer_sizes=(128, 128),
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-3,
        max_epochs=500,
        patience=10,
        validation_fraction=0.10,
        random_state=0,
        verbose=True
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, input_dim):
        layers = []
        prev = input_dim
        for size in self.hidden_layer_sizes:
            layers.append(
                nn.Linear(prev, size)
            )
            layers.append(
                nn.ReLU()
            )
            prev = size

        layers.append(
            nn.Linear(prev, 1)
        )

        return nn.Sequential(*layers)


    def fit(self, X, y, groups=None):
        torch.manual_seed(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        if groups is None:
            raise ValueError(
                "groups must be provided for group-aware early stopping"
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.validation_fraction,
            random_state=self.random_state
        )

        train_idx, stop_idx = next(
            splitter.split(
                X,
                y,
                groups
            )
        )

        X_train = X[train_idx]
        X_stop = X[stop_idx]

        y_train = y[train_idx]
        y_stop = y[stop_idx]

        X_train = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        y_train = torch.tensor(
            y_train,
            dtype=torch.float32
        )

        X_stop = torch.tensor(
            X_stop,
            dtype=torch.float32
        )

        y_stop = torch.tensor(
            y_stop,
            dtype=torch.float32
        )

        self.model_ = self._build_model(
            X_train.shape[1]
        )

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        loss_fn = nn.MSELoss()

        best_loss = np.inf
        best_state = None
        wait = 0

        self.history_ = []

        for epoch in range(self.max_epochs):
            self.model_.train()

            permutation = torch.randperm(
                len(X_train)
            )

            for i in range(
                0,
                len(X_train),
                self.batch_size
            ):

                idx = permutation[
                    i:i+self.batch_size
                ]

                pred = self.model_(
                    X_train[idx]
                )

                loss = loss_fn(
                    pred,
                    y_train[idx]
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            # validation loss
            self.model_.eval()

            with torch.no_grad():
                stop_pred = self.model_(
                    X_stop
                )

                stop_loss = loss_fn(
                    stop_pred,
                    y_stop
                ).item()

            self.history_.append(stop_loss)

            if self.verbose:
                print(
                    f"epoch={epoch}, val_loss={stop_loss:.5f}"
                )

            if stop_loss < best_loss:

                best_loss = stop_loss
                wait = 0

                best_state = copy.deepcopy(
                    self.model_.state_dict()
                )

            else:
                wait += 1

            if wait >= self.patience:
                break

        self.n_epochs_ = epoch + 1

        self.model_.load_state_dict(
            best_state
        )

        return self


    def predict(self, X):
        X = np.asarray(X)

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.model_.eval()

        with torch.no_grad():
            pred = self.model_(
                X
            ).numpy()

        return pred.ravel()