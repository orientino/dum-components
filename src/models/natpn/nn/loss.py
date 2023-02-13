from typing import Literal
import torch
from torch import nn
from src.models.natpn.distributions import Posterior

Reduction = Literal["mean", "sum", "none"]


class BayesianLoss(nn.Module):
    """
    The Bayesian loss computes an uncertainty-aware loss based on the parameters of a conjugate
    prior of the target distribution.
    """

    def __init__(self, entropy_weight: float = 0.0, reconst_weight: float = 0.0, reduction: Reduction = "mean"):
        """
        Args:
            entropy_weight: The weight for the entropy regualarizer.
            reconst_weight: TODO
            reduction: The reduction to apply to the loss. Must be one of "mean", "sum", "none".
        """
        super().__init__()
        self.entropy_weight = entropy_weight
        self.reconst_weight = reconst_weight
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=self.reduction)

    def forward(self, y_pred: Posterior, y_true: torch.Tensor, X: torch.Tensor = None, y_hat: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the loss of the prediction with respect to the target.

        Args:
            y_pred: The posterior distribution predicted by the Natural Posterior Network.
            y_true: The true targets. Either indices for classes of a classification problem or
                the real target values. Must have the same batch shape as ``y_pred``.
            X: TODO
            y_hat: TODO

        Returns:
            The loss, processed according to ``self.reduction``.
        """
        nll = -y_pred.expected_log_likelihood(y_true)
        loss = nll - self.entropy_weight * y_pred.entropy()

        if self.reduction == "mean":
            loss = loss.mean()
            nll = nll.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            nll = nll.mean()

        # Add the reconstruction loss
        mse = self.mse(X, y_hat) if X != None else 0
        loss = loss + self.reconst_weight * mse

        return loss, nll, mse
