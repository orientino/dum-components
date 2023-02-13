from typing import Tuple, List
import torch
from torch import nn
import src.models.natpn.distributions as D
from .flow import NormalizingFlow
from .output import Output
from .scaler import CertaintyBudget, EvidenceScaler


class NaturalPosteriorNetworkModel(nn.Module):
    """
    Implementation of the NatPN module. This class only describes the forward pass through the
    model and can be compiled via TorchScript.
    """

    def __init__(
        self,
        latent_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        flows: nn.ModuleList,
        output: Output,
        budget: CertaintyBudget = "normal",
    ):
        """
        Args:
            latent_dim: The dimension of the latent space to which the model's encoder maps.
            config: The model's intrinsic configuration.
            encoder: The model's (deep) encoder which maps input to a latent space.
            decoder: The model's decoder to reconstruct the input.
            flows: The list of model's normalizing flow which yields the evidence of inputs based on their
                latent representations.
            output: The model's output head which maps each input's latent representation linearly
                to the parameters of the target distribution.
            budget: The scaling factor for the certainty budget that the normalizing
                flow can draw from.
            budge_coeff: TODO
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.flows = flows
        self.output = output
        self.scaler = EvidenceScaler(latent_dim, budget)

    def forward(self, x: torch.Tensor) -> Tuple[D.Posterior, torch.Tensor]:
        """
        Performs a Bayesian update over the target distribution for each input independently. The
        returned posterior distribution carries all information about the prediction.

        Args:
            x: The inputs that are first passed to the encoder.

        Returns:
            The posterior distribution for every input along with their log-probabilities. The
            same probabilities are returned from :meth:`log_prob`.
        """
        if self.decoder and self.encoder.training: 
            update, log_prob, y_hat = self.posterior_update(x)
            return self.output.prior.update(update), log_prob, y_hat
        else:
            update, log_prob = self.posterior_update(x)
            return self.output.prior.update(update), log_prob

    def posterior_update(self, x: torch.Tensor) -> Tuple[D.PosteriorUpdate, torch.Tensor]:
        """
        Computes the posterior update over the target distribution for each input independently.

        Args:
            x: The inputs that are first passed to the encoder.

        Returns:
            The posterior update for every input and the true log-probabilities.
        """
        if self.decoder and self.encoder.training:
            z, r = self.encoder.forward(x)
            y_hat = self.decoder.forward(r) 
        else:
            z = self.encoder.forward(x)

        if z.dim() > 2:
            z = z.permute(0, 2, 3, 1)
        prediction = self.output.forward(z)
        sufficient_statistics = prediction.expected_sufficient_statistics()

        # divide the latent_dim into chunks and assign each chunk to a flow
        z = z.split(z.shape[1] // len(self.flows), -1)
        log_prob = [flow.forward(z_i) for flow, z_i in zip(self.flows, z)]
        log_prob = sum(log_prob)
        log_evidence = self.scaler.forward(log_prob)

        # reconstruction input
        if self.decoder and self.encoder.training:
            return D.PosteriorUpdate(sufficient_statistics, log_evidence), log_prob, y_hat
        else:
            return D.PosteriorUpdate(sufficient_statistics, log_evidence), log_prob

    def log_prob(self, x: torch.Tensor, track_encoder_gradients: bool = True) -> torch.Tensor:
        """
        Computes the (scaled) log-probability of observing the given inputs.

        Args:
            x: The inputs that are first passed to the encoder.
            track_encoder_gradients: Whether to track the gradients of the encoder.

        Returns:
            The per-input log-probability.
        """
        with torch.set_grad_enabled(self.training and track_encoder_gradients):
            if self.decoder:
                z, _ = self.encoder.forward(x)
            else:
                z = self.encoder.forward(x)

            if z.dim() > 2:
                z = z.permute(0, 2, 3, 1)

        # divide the latent_dim into chunks and assign each chunk to a flow
        z = z.split(z.shape[1] // len(self.flows), -1)
        log_prob = sum([flow.forward(z_i) for flow, z_i in zip(self.flows, z)])

        return log_prob
