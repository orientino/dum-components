import math
from typing import Literal
import torch
from torch import nn

CertaintyBudget = Literal["constant", "linear-exp", "linear-log", "normal"]
"""
The certainty budget to distribute in the latent space of dimension ``H``:

- ``constant``: A certainty budget of 1, independent of the latent space's dimension.
- ``linear-exp``: A certainty budget of ``exp(coeff * H)``.
- ``linear-log``: A certainty budget of ``coeff * H``.
- ``normal``: A certainty budget that causes a multivariate normal distribution to yield the same
  probability at the origin at any dimension: ``exp(0.5 * log(4 * pi) * H)``.
"""


class EvidenceScaler(nn.Module):
    """
    Scaler for the evidence to distribute a certainty budget other than one in the latent space.
    """

    def __init__(self, dim: int, budget: CertaintyBudget):
        """
        Args:
            dim: The dimension of the latent space.
            budget: The budget to use.
        """
        super().__init__()
        if budget == "normal":
            self.log_scale = 0.5 * math.log(4 * math.pi) * dim
        elif budget == "log-exp":
            self.log_scale = 0.5 * math.log(dim) * dim
        elif budget == "constant":
            self.log_scale = 0
        else:
            raise NotImplementedError

    def forward(self, log_evidence: torch.Tensor) -> torch.Tensor:
        """
        Scales the evidence in the log space according to the certainty budget.

        Args:
            log_evidence: The log-evidence to scale.

        Returns:
            The scaled and clamped evidence in the log-space.
        """
        return self._clamp_preserve_gradients(log_evidence + self.log_scale, lower=-30.0, upper=30.0)

    def _clamp_preserve_gradients(self, x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        """
        Clamps the values of the tensor into ``[lower, upper]`` but keeps the gradients.

        Args:
            x: The tensor whose values to constrain.
            lower: The lower limit for the values.
            upper: The upper limit for the values.

        Returns:
            The clamped tensor.
        """
        return x + (x.clamp(min=lower, max=upper) - x).detach()
