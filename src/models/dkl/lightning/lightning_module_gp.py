# pylint: disable=abstract-method
from typing import Any, Optional, cast, Dict, List, Tuple, Union
import pytorch_lightning as pl
import torch
import gpytorch
from src.models.dkl.lightning.lightning_module import DeepKernelLearningLightningModule

Batch = Tuple[torch.Tensor, torch.Tensor]


class DeepKernelLearningGPLightningModule(DeepKernelLearningLightningModule):
    """
    Lightning module for optimizing the Gaussian Process of DUE.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        learning_rate_gp: float = 1e-3,
        early_stopping: bool = True,
        optim: str = "sgd",
        scheduler: str = "cosine5e-4",
        n_batches: int = 1,
        max_epochs: int = 1,
        phase: str = "train",
    ):
        super().__init__(
            model,
            learning_rate,
            learning_rate_gp,
            early_stopping,
            optim,
            scheduler,
            n_batches,
            max_epochs,
            phase,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        # Initialize optimizer
        models = [
            {"params": self.model.gp.hyperparameters()},
            {"params": self.model.gp.variational_parameters()},
            {"params": self.model.likelihood.parameters()},
        ]
        optim = torch.optim.AdamW(
            models,
            lr=self.learning_rate_gp,
            weight_decay=1e-6,
        )

        return optim

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        X, y_true = batch
        if self.model.reconst_weight > 0 and self.model.encoder.training:
            y_pred, mse = self(X)
            elbo = self.model.loss(y_pred, y_true)
            loss = elbo + mse
            self.log(f"train/mse_{self.phase}", mse)
            self.log(f"train/elbo_{self.phase}", elbo)
        else:
            y_pred = self(X)
            loss = self.model.loss(y_pred, y_true)

        self.log(f"train/loss_{self.phase}", loss, prog_bar=True)
        return loss
