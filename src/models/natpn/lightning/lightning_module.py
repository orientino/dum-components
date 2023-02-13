# pylint: disable=abstract-method
from typing import Any, cast, Dict, List, Tuple
import pytorch_lightning as pl
import torch
from torch.nn import ModuleList
from torchmetrics import Accuracy, MeanSquaredError, AUROC
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import src.models.natpn.distributions as D
from src.datasets import DataModule
from src.metrics import BrierScore, QuantileCalibrationScore
from src.models.natpn.nn import BayesianLoss, NaturalPosteriorNetworkModel
from src.models.natpn.nn.output import CategoricalOutput

Batch = Tuple[torch.Tensor, torch.Tensor]


class NaturalPosteriorNetworkLightningModule(pl.LightningModule):
    """
    Lightning module for training and evaluating NatPN.
    """

    def __init__(
        self,
        model: NaturalPosteriorNetworkModel,
        learning_rate: float = 1e-3,
        learning_rate_nf: float = 1e-3,
        entropy_weight: float = 0.0,
        reconst_weight: float = 0.0,
        early_stopping: bool = False,
        optim: str = "adam",
        scheduler: str = "multistep",
        n_batches: int = 1,
    ):
        """
        Args:
            model: The model to train or evaluate. If training, this *must* be a
                :class:`NaturalPosteriorNetworkModel`.
            learning_rate: The learning rate to use for the Adam optimizer.
            entropy_weight: The weight of the entropy regularizer in the Bayesian loss.
        """
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.learning_rate_nf = learning_rate_nf
        self.loss = BayesianLoss(entropy_weight, reconst_weight)
        self.early_stopping = early_stopping
        self.optim = optim
        self.scheduler = scheduler
        self.n_batches = n_batches

        self.data2dist = {0: "id", 1: "ood", 2: "oodom"}
        self.alea_conf_roc = AUROC(dist_sync_fn=self.all_gather)
        self.epist_conf_roc = AUROC(dist_sync_fn=self.all_gather)

        if isinstance(model.output, CategoricalOutput):
            # We have discrete output
            self.output = "discrete"
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
        else:
            # We have continuous output
            self.output = "continuous"
            self.rmse = ModuleList([
                MeanSquaredError(squared=False, compute_on_step=False, dist_sync_fn=self.all_gather),
                MeanSquaredError(squared=False, compute_on_step=False, dist_sync_fn=self.all_gather),
                MeanSquaredError(squared=False, compute_on_step=False, dist_sync_fn=self.all_gather),
            ])
            self.calibration = ModuleList([
                QuantileCalibrationScore(compute_on_step=False, dist_sync_fn=self.all_gather),
                QuantileCalibrationScore(compute_on_step=False, dist_sync_fn=self.all_gather),
                QuantileCalibrationScore(compute_on_step=False, dist_sync_fn=self.all_gather),
            ])

    def configure_optimizers(self) -> Dict[str, Any]:
        # Initialize optimizer
        models1 = [
            {"params": self.model.encoder.parameters()},
            {"params": self.model.output.parameters()},
        ]
        models2 = [
            {"params": self.model.flows.parameters()},
        ]
        if self.model.decoder:
            models1 += [{"params": self.model.decoder.parameters()}]

        if self.optim == "adam":
            optim = [torch.optim.Adam(
                models1,
                lr=self.learning_rate,
            )]
        elif self.optim == "adamw":
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
                lr=self.learning_rate_nf,
                weight_decay=1e-6,
            )
        )

        # Initialize scheduler
        if self.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim[0],
                    factor=0.2,
                    patience=self.trainer.max_epochs // 5,
                    min_lr=1e-7,
                ),
                "monitor": "val/id/loss",
            }
        elif self.scheduler == "multistep":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim[0], milestones=[60, 120, 160], gamma=0.2
                )
            }
        elif self.scheduler == "cosine5e-4":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[0],
                    T_max=self.trainer.max_epochs * self.n_batches,
                    eta_min=5e-4,
                ),
                "interval": "step",
            }
        elif self.scheduler == "cosine1e-4":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[0],
                    T_max=self.trainer.max_epochs * self.n_batches,
                    eta_min=1e-4,
                ),
                "interval": "step",
            }
        elif self.scheduler == "cosine5e-5":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[0],
                    T_max=self.trainer.max_epochs * self.n_batches,
                    eta_min=5e-5,
                ),
                "interval": "step",
            }
        elif self.scheduler == "cosine1e-5":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[0],
                    T_max=self.trainer.max_epochs * self.n_batches,
                    eta_min=1e-5,
                ),
                "interval": "step",
            }
        elif self.scheduler == "cosine1e-6":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[0],
                    T_max=self.trainer.max_epochs * self.n_batches,
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
            EarlyStopping("val/id/loss", min_delta=1e-3, patience=10),
        ]

    def forward(self, X):
        # Return with / without reconstructed input
        if self.model.decoder and self.model.encoder.training:
            y_pred, log_prob, y_hat = self.model.forward(X)
            return y_pred, log_prob, y_hat
        else:
            y_pred, log_prob = self.model.forward(X)
            return y_pred, log_prob

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        X, y_true = batch

        # Bayesian loss with / without reconstruction error
        if self.model.decoder:
            y_pred, log_prob, y_hat = self(X)
            loss, nll, mse = self.loss.forward(y_pred, y_true, X, y_hat)
            self.log("train/nll", nll, prog_bar=True)
            self.log("train/mse", mse, prog_bar=True)
        else:
            y_pred, log_prob = self(X)
            loss, _, _ = self.loss.forward(y_pred, y_true)
            # loss = torch.nn.functional.nll_loss(y_pred.maximum_a_posteriori().logits, y_true)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/log_prob", log_prob.mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        X, y_true = batch
        y_pred, log_prob = self(X)
        loss, _, _ = self.loss.forward(y_pred, y_true)
        # loss = torch.nn.functional.nll_loss(y_pred.maximum_a_posteriori().logits, y_true)

        prefix = f"val/{self.data2dist[dataloader_idx]}"
        self.log(f"{prefix}/loss", loss, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{prefix}/log_prob", log_prob.mean(), add_dataloader_idx=False)
        self._compute_metrics(prefix, y_pred, y_true, dataloader_idx)

    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        X, y_true = batch
        y_pred, log_prob = self(X)
        loss, _, _ = self.loss.forward(y_pred, y_true)
        # loss = torch.nn.functional.nll_loss(y_pred.maximum_a_posteriori().logits, y_true)

        prefix = f"test/{self.data2dist[dataloader_idx]}"
        prefix = f"test/ood{dataloader_idx}" if dataloader_idx > 0 else prefix
        self.log(f"{prefix}/loss", loss, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{prefix}/log_prob", log_prob.mean(), add_dataloader_idx=False)
        self._compute_metrics(prefix, y_pred, y_true, dataloader_idx)

    def _compute_metrics(self, prefix: str, y_pred: D.Posterior, y_true: torch.Tensor, dataloader_idx: int = 0) -> None:
        if self.output == "discrete":
            self.accuracy[dataloader_idx].update(y_pred.maximum_a_posteriori().mean(), y_true)
            self.log(f"{prefix}/accuracy", self.accuracy[dataloader_idx], prog_bar=True, add_dataloader_idx=False)

            probs = y_pred.maximum_a_posteriori().expected_sufficient_statistics()
            self.brier_score[dataloader_idx].update(probs, y_true)
            self.log(f"{prefix}/brier_score", self.brier_score[dataloader_idx], add_dataloader_idx=False)
        else:
            dm = cast(DataModule, self.trainer.datamodule)
            predicted = y_pred.maximum_a_posteriori().mean()
            self.rmse[dataloader_idx].update(dm.transform_output(predicted), dm.transform_output(y_true))
            self.log(f"{prefix}/rmse", self.rmse[dataloader_idx], prog_bar=True, add_dataloader_idx=False)

            confidence_levels = y_pred.posterior_predictive().symmetric_confidence_level(y_true)
            self.calibration[dataloader_idx].update(confidence_levels)
            self.log(f"{prefix}/calibration", self.calibration[dataloader_idx], add_dataloader_idx=False)
