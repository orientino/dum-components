# pylint: disable=abstract-method
from typing import Any, Optional, cast, Dict, List, Tuple, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
import gpytorch
from torch.nn import ModuleList
from torchmetrics import Accuracy, AUROC
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.architectures.encoder import *
from src.metrics import BrierScore

Batch = Tuple[torch.Tensor, torch.Tensor]


class DeepKernelLearningLightningModule(pl.LightningModule):
    """
    Lightning module for training and evaluating DUE.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        learning_rate_head: float = 1e-3,
        early_stopping: bool = False,
        optim: str = "sgd",
        scheduler: str = "cosine5e-4",
        n_batches: int = 1,
        max_epochs: int = 1,
        phase: str = "train",
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.learning_rate_head = learning_rate_head
        self.early_stopping = early_stopping
        self.optim = optim
        self.scheduler = scheduler
        self.n_batches = n_batches
        self.max_epochs = max_epochs
        self.phase = phase
        self.data2dist = {0: "id", 1: "ood", 2: "oodom"}
        self.pred_conf_roc = AUROC(dist_sync_fn=self.all_gather)
        self.mse = nn.MSELoss()

        self.accuracy = ModuleList([
            Accuracy(compute_on_step=False, dist_sync_fn=self.all_gather),
            Accuracy(compute_on_step=False, dist_sync_fn=self.all_gather),
            Accuracy(compute_on_step=False, dist_sync_fn=self.all_gather),
        ])
        self.brier_score = ModuleList([
            BrierScore(compute_on_step=False, dist_sync_fn=self.all_gather),
            BrierScore(compute_on_step=False, dist_sync_fn=self.all_gather),
            BrierScore(compute_on_step=False, dist_sync_fn=self.all_gather),
        ])

        assert phase in ["train", "finetune"]

    def configure_optimizers(self) -> Dict[str, Any]:
        # Initialize optimizer
        models1 = [
            {"params": self.model.encoder.parameters()},
        ]
        models2 = [
            {"params": self.model.gp.hyperparameters()},
            {"params": self.model.gp.variational_parameters()},
            {"params": self.model.likelihood.parameters()},
        ]
        if self.model.reconst_weight > 0:
            models1 += [{"params": self.model.decoder.parameters()}]

        if self.optim == "adamw":
            optim = [torch.optim.AdamW(
                models1,
                lr=self.learning_rate,
                weight_decay=1e-6,
            )]
        elif self.optim == "sgd":
            optim = [torch.optim.SGD(
                models1,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )]
        else:
            raise NotImplementedError

        optim.append(
            torch.optim.AdamW(
                models2,
                lr=self.learning_rate_head,
                weight_decay=1e-6,
            )
        )

        # Initialize scheduler
        if self.scheduler == "multistep":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim[0], milestones=[60, 120, 160], gamma=0.2
                )
            }
        elif self.scheduler == "cosine5e-4":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[0],
                    T_max=self.max_epochs * self.n_batches,
                    eta_min=5e-4,
                ),
                "interval": "step",
            }
        elif self.scheduler == "cosine1e-5":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[0],
                    T_max=self.max_epochs * self.n_batches,
                    eta_min=1e-5,
                ),
                "interval": "step",
            }
        elif self.scheduler == "cosine1e-6":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[0],
                    T_max=self.max_epochs * self.n_batches,
                    eta_min=1e-6,
                ),
                "interval": "step",
            }
        else:
            raise NotImplementedError

        return optim, lr_scheduler

    def configure_callbacks(self) -> List[pl.Callback]:
        if not self.early_stopping:
            return []
        return [
            EarlyStopping(f"val/id/loss_{self.phase}", min_delta=1e-3, patience=10),
        ]

    def forward(self, X):
        # Return with / without reconstructed input
        if self.model.reconst_weight > 0 and self.model.encoder.training:
            y_pred, y_hat = self.model.forward(X)
            return y_pred, y_hat
        else:
            y_pred = self.model.forward(X)
            return y_pred

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        X, y_true = batch
        if self.model.reconst_weight > 0 and self.model.encoder.training:
            y_pred, y_hat = self(X)
            elbo = self.model.loss(y_pred, y_true)
            mse = self.model.reconst_weight * self.mse(X, y_hat)
            loss = elbo + mse
            self.log(f"train/mse_{self.phase}", mse, prog_bar=True)
            self.log(f"train/elbo_{self.phase}", elbo, prog_bar=True)
        else:
            y_pred = self(X)
            loss = self.model.loss(y_pred, y_true)

        self.log(f"train/loss_{self.phase}", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        X, y_true = batch
        y_pred = self(X)
        loss = self.model.loss(y_pred, y_true)

        prefix = f"val/{self.data2dist[dataloader_idx]}"
        self.log(f"{prefix}/loss_{self.phase}", loss, prog_bar=True, add_dataloader_idx=False)
        self._compute_metrics(prefix, y_pred, y_true, dataloader_idx)

    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        X, y_true = batch
        y_pred = self(X)
        loss = self.model.loss(y_pred, y_true)

        prefix = f"test/{self.data2dist[dataloader_idx]}"
        prefix = f"test/ood{dataloader_idx}" if dataloader_idx > 0 else prefix
        self.log(f"{prefix}/loss", loss, prog_bar=True, add_dataloader_idx=False)
        self._compute_metrics(prefix, y_pred, y_true, dataloader_idx)

    def _compute_metrics(self, prefix: str, y_pred, y_true, dataloader_idx: int = 0) -> None:
        # Sample softmax values independently for classification at test time
        # The mean here is over likelihood samples
        with gpytorch.settings.num_likelihood_samples(32):
            y_pred = y_pred.to_data_independent_dist()
            output = self.model.likelihood(y_pred).probs.mean(0)
            y_pred = torch.argmax(output, dim=1)

        self.accuracy[dataloader_idx].update(y_pred, y_true)
        self.log(f"{prefix}/accuracy_{self.phase}", self.accuracy[dataloader_idx], prog_bar=True, add_dataloader_idx=False)

        self.brier_score[dataloader_idx].update(output, y_true)
        self.log(f"{prefix}/brier_score_{self.phase}", self.brier_score[dataloader_idx], prog_bar=True, add_dataloader_idx=False)
