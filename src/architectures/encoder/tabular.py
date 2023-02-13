import torch
from typing import List, Tuple
from torch import nn
from src.architectures.activation import *
from src.architectures.spectral.spectral_norm_fc import spectral_norm_fc
from src.architectures.spectral.spectral_batchnorm import SpectralBatchNorm1d


class TabularEncoder(nn.Module):
    """
    Encoder for tabular data with bi-Lipschitz property using residual connections and spectral normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        *,
        act: ActivationType = "relu",
        residual: bool = False,
        spectral: Tuple[bool, bool, bool] = (False, False, False),      # (spectral_fc, spectral_conv, spectral_bn)
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        bn_out: bool = False,
        reconst_out: bool = False,
    ):
        """
        Args:
            input_dim: The dimension of the inputs.
            hidden_dims: The dimensions of the hidden layers.
            output_dim: The dimension of the output, i.e. the latent space.
            spectral: Use the spectral normalization 
            residual: Use residual shortcut connection
            coeff: The lipschitz constant
            n_power_iterations: Number of iterations for the spectral normalization
            bn_out: Add batch normalization to the output
            reconst_out: Act as the decoder: forward() will output a single value instead of a tuple
        """
        super(TabularEncoder, self).__init__()

        def wrapper_fc(in_dim, out_dim):
            if spectral[0]:
                return spectral_norm_fc(nn.Linear(in_dim, out_dim), coeff, n_power_iterations)
            return nn.Linear(in_dim, out_dim)

        def wrapped_bn(num_features):
            if spectral[2]:
                return SpectralBatchNorm1d(num_features, coeff) 
            return nn.BatchNorm1d(num_features)
            
        # Define the activation function
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == "mish":
            self.act = nn.Mish()
        elif act == "silu":
            self.act = nn.SiLU()
        elif act == "l-matern":
            self.act = matern
        elif act == "p-sin":
            self.act = sin
        elif act == "p-sincos":
            self.act = sincos
        elif act == "p-triangle":
            self.act = triangle
        elif act == "p-relu":
            self.act = periodic_relu
        else:
            raise NotImplementedError

        # Define the architecture
        self.linear_in = wrapper_fc(input_dim, hidden_dims[0])
        self.layers = nn.ModuleList([
            _TabularBlock(wrapper_fc, hidden_dims[0], self.act, residual) for _ in hidden_dims
        ])
        self.linear_out = wrapper_fc(hidden_dims[0], output_dim)

        # Add BN at the end to stabilize the training: It facilitates the match between the 
        # latent positions output by the encoder and non-zero density regions learned by the normalizing flows. 
        self.bn_out = wrapped_bn(output_dim) if bn_out else None

        # Is this used as a decoder
        self.reconst_out = reconst_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.linear_out(out)

        # Add BN to the output
        if self.bn_out:
            out = self.bn_out(out)

        # Return the input for the reconstruction decoder
        if self.reconst_out:
            return out, rec

        return out


class _TabularBlock(nn.Module):
    """
    Linear layer with bi-Lipschitz property using residual connections and spectral normalization.
    """

    def __init__(self, wrapper_fc, dim: int, act: ActivationType, residual: bool = False):
        super(_TabularBlock, self).__init__()
        self.linear = wrapper_fc(dim, dim)
        self.act = act
        self.residual = residual


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.linear(x)
        if self.residual:
            out += identity
        out = self.act(out)

        return out
