from ._base import NormalizingFlow
import torch
import torch.nn as nn
import normflows as nf
import larsflow as lf


class NeuralSplineFlow(nn.Module):
    """
    Neural Spline Flow with resampled base.
    https://github.com/VincentStimper/resampled-base-flows
    """

    def __init__(
        self, 
        dim: int, 
        num_layers: int = 8,
        hidden_units: int = 128,
        hidden_layers: int = 2,
        resampled: bool = False,
    ):

        """
        dim: The input dimension of the normalizing flow.
        """
        super().__init__()

        flows = []
        for i in range(num_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(dim, hidden_layers, hidden_units)]
            flows += [nf.flows.LULinearPermute(dim)]

        # Set prior and q0
        if resampled:
            a = nf.nets.MLP([dim, 256, 256, 1], output_fn="sigmoid")
            self.q0 = lf.distributions.ResampledGaussian(dim, a, 100, 0.1, trainable=False)
        else:
            self.q0 = nf.distributions.DiagGaussian(dim, trainable=False)
            
        # Construct flow model
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        """
        Get log probability for batch
        :param x: Batch
        :return: log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows)-1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q
