# pylint: disable=abstract-method
import pytorch_lightning as pl
import gpytorch
import torch
from torchmetrics import Accuracy, AUROC
from src.metrics import BrierScore
from .lightning_module import Batch


class DeepKernelLearningOodTestingLightningModule(pl.LightningModule):
    """
    Lightning module for evaluating the OOD detection performance of DUE.
    """

    def __init__(
        self,
        model,
        logging_key: str,
    ):
        super().__init__()
        self.model = model
        self.logging_key = logging_key

        self.pred_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

    def test_step(self, batch: Batch, _batch_idx: int) -> None:
        X, y_true = batch
        y_pred = self.model.forward(X)

        # Sample softmax values independently for classification at test time
        # The mean here is over likelihood samples
        with gpytorch.settings.num_likelihood_samples(32):
            y_pred = y_pred.to_data_independent_dist()
            output = self.model.likelihood(y_pred).probs.mean(0)

        uncertainty = -(output * output.log()).sum(1)
        uncertainty = 1 - uncertainty

        self.pred_conf_roc.update(uncertainty, y_true)
        self.log(f"{self.logging_key}/predictive_confidence_auc_roc", self.pred_conf_roc)


class DeepKernelLearningOodCorruptedLightningModule(pl.LightningModule):
    """
    Lightning module for evaluating the performance of DUE on the CIFAR corrupted dataset.
    """

    def __init__(
        self,
        model,
        logging_key,
    ):
        """
        Args:
            model: The model to train or evaluate. 
        """
        super().__init__()
        self.model = model
        self.logging_key = logging_key
        self.accuracy = Accuracy(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.brier_score = BrierScore(compute_on_step=False, dist_sync_fn=self.all_gather)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        X, y_true = batch
        y_pred = self.model.forward(X)
        loss = self.model.loss(y_pred, y_true)

        prefix = f"test/{self.logging_key}"
        self.log(f"{prefix}/loss", loss, prog_bar=True)
        self._compute_metrics(prefix, y_pred, y_true)

    def _compute_metrics(self, prefix: str, y_pred, y_true) -> None:
        # Sample softmax values independently for classification at test time
        # The mean here is over likelihood samples
        with gpytorch.settings.num_likelihood_samples(32):
            y_pred = y_pred.to_data_independent_dist()
            output = self.model.likelihood(y_pred).probs.mean(0)
            y_pred = torch.argmax(output, dim=1)

        self.accuracy.update(y_pred, y_true)
        self.log(f"{prefix}/accuracy", self.accuracy, prog_bar=True)

        self.brier_score.update(output, y_true)
        self.log(f"{prefix}/brier_score", self.brier_score, prog_bar=True)
