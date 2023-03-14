# pylint: disable=abstract-method
from typing import Any, Dict, List
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import optim
from torch.nn import ModuleList
from torchmetrics import Accuracy, MeanSquaredError
from src.metrics import BrierScore, QuantileCalibrationScore
from src.models.natpn.nn import NaturalPosteriorNetworkModel
from src.models.natpn.nn.output import CategoricalOutput
from .lightning_module import Batch


class NaturalPosteriorNetworkFlowLightningModule(pl.LightningModule):
    """
    Lightning module for optimizing the normalizing flow of NatPN.
    """

    def __init__(
        self,
        model: NaturalPosteriorNetworkModel,
        learning_rate: float = 1e-3,
        early_stopping: bool = False,
        phase: str = "train",
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.phase = phase
        self.data2dist = {0: "id", 1: "ood", 2: "oodom"}

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

        assert phase in ["warmup", "train", "finetune"]

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.AdamW(
            self.model.flows.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-6,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[
                    0.7 * self.trainer.max_epochs, 
                    0.9 * self.trainer.max_epochs,
                ], gamma=0.2
            )
        }

        if self.phase == "warmup":
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=(lambda x: (x+1) / self.trainer.max_epochs),
                )
            }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self) -> List[pl.Callback]:
        if not self.early_stopping:
            return []
        return [
            EarlyStopping(
                f"val/id/log_prob_{self.phase}",
                min_delta=1e-3,
                mode="max",
                patience=self.trainer.max_epochs // 10,
            ),
        ]

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        X, _ = batch
        log_prob = self.model.log_prob(X, track_encoder_gradients=False).mean()
        self.log(f"train/log_prob_{self.phase}", log_prob, prog_bar=True)
        return -log_prob

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        X, y_true = batch
        y_pred, log_prob = self.model.forward(X)

        prefix = f"val/{self.data2dist[dataloader_idx]}"
        self.log(f"{prefix}/log_prob_{self.phase}", log_prob.mean(), add_dataloader_idx=False)

        # Compute metrics
        if self.output == "discrete":
            self.accuracy[dataloader_idx].update(y_pred.maximum_a_posteriori().mean(), y_true)
            self.log(f"{prefix}/accuracy_{self.phase}", self.accuracy[dataloader_idx], prog_bar=True, add_dataloader_idx=False)

            probs = y_pred.maximum_a_posteriori().expected_sufficient_statistics()
            self.brier_score[dataloader_idx].update(probs, y_true)
            self.log(f"{prefix}/brier_score_{self.phase}", self.brier_score[dataloader_idx], add_dataloader_idx=False)
        else:
            dm = self.trainer.datamodule
            predicted = y_pred.maximum_a_posteriori().mean()
            self.rmse[dataloader_idx].update(dm.transform_output(predicted), dm.transform_output(y_true))
            self.log(f"{prefix}/rmse_{self.phase}", self.rmse[dataloader_idx], prog_bar=True, add_dataloader_idx=False)

            confidence_levels = y_pred.posterior_predictive().symmetric_confidence_level(y_true)
            self.calibration[dataloader_idx].update(confidence_levels)
            self.log(f"{prefix}/calibration_{self.phase}", self.calibration[dataloader_idx], add_dataloader_idx=False)
