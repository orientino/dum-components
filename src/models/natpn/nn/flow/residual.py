from ._base import NormalizingFlow
import torch
import torch.nn as nn
import normflow as nf


class ResidualFlow(nn.Module):
    """
    Normalizing flow that consists purely of a series of radial transforms.
    """

    def __init__(
        self, 
        dim: int, 
        num_layers: int = 8,
        hidden_units: int = 128,
        hidden_layers: int = 3,
    ):

        """
        dim: The input dimension of the normalizing flow.
        """
        super().__init__()

        flows = []
        for i in range(num_layers):
            net = nf.nets.LipschitzMLP([dim] + [hidden_units] * (hidden_layers - 1) + [dim], init_zeros=True, lipschitz_const=0.9)
            flows += [nf.flows.Residual(net, reduce_memory=True)]
            flows += [nf.flows.ActNorm(dim)]

        # Set prior and q0
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
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q
