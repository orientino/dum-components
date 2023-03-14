from __future__ import annotations
import torch
import numpy as np


class StandardScaler:
    """
    Standard scaler for tabular data.
    """

    mean_: torch.Tensor
    std_: torch.Tensor

    def fit(self, X: torch.Tensor) -> StandardScaler:
        """
        Fits the mean and std of this scaler via the provided tabular data.
        """
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transforms the provided tabular data with the mean and std that was fitted previously.
        """
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform of tabular data.
        """
        return X * self.std_ + self.mean_


class IdentityScaler:
    """
    A scaler which does nothing.
    """

    def fit(self, _X: torch.Tensor) -> IdentityScaler:
        """
        Noop.
        """
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Identity.
        """
        return X

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Identity.
        """
        return X


def scale_oodom(x: torch.Tensor) -> torch.Tensor:
    """
    Scales the given input with a constant of 255 such that it can be considered out-of-domain.
    """
    return x * 255


def zero_channel(x: torch.Tensor) -> torch.Tensor:
    """
    Zeroes a channel
    """
    x[np.random.randint(0, 3), :, :] *= 0
    return x
