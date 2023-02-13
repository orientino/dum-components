# pylint: disable=abstract-method
from typing import Union
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from src.metrics import BrierScore, AUCPR
import src.models.natpn.distributions as D
from src.models.natpn.nn import BayesianLoss, NaturalPosteriorNetworkModel
from .lightning_module import Batch


class NaturalPosteriorNetworkOodTestingLightningModule(pl.LightningModule):
    """
    Lightning module for evaluating the OOD detection performance of NatPN.
    """

    def __init__(
        self,
        model: NaturalPosteriorNetworkModel,
        logging_key: str,
    ):
        super().__init__()
        self.model = model
        self.logging_key = logging_key

        self.alea_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.alea_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        self.epist_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.epist_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        self.pred_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.pred_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

    def test_step(self, batch: Batch, _batch_idx: int) -> None:
        X, y = batch
        posterior, log_prob = self.model.forward(X)

        self.log(f"{self.logging_key}/log_prob", log_prob)

        # Aleatoric confidence (from negative uncertainty)
        aleatoric_conf = -posterior.maximum_a_posteriori().uncertainty()
        if aleatoric_conf.dim() > 1:
            aleatoric_conf = aleatoric_conf.mean(tuple(range(1, aleatoric_conf.dim())))
        self.alea_conf_pr.update(aleatoric_conf, y)
        self.log(f"{self.logging_key}/aleatoric_confidence_auc_pr", self.alea_conf_pr)
        self.alea_conf_roc.update(aleatoric_conf, y)
        self.log(f"{self.logging_key}/aleatoric_confidence_auc_roc", self.alea_conf_roc)

        # Epistemic confidence
        epistemic_conf = log_prob
        if epistemic_conf.dim() > 1:
            epistemic_conf = epistemic_conf.mean(tuple(range(1, epistemic_conf.dim())))
        self.epist_conf_pr.update(epistemic_conf, y)
        self.log(f"{self.logging_key}/epistemic_confidence_auc_pr", self.epist_conf_pr)
        self.epist_conf_roc.update(epistemic_conf, y)
        self.log(f"{self.logging_key}/epistemic_confidence_auc_roc", self.epist_conf_roc)

        # Predictive confidence (from negative uncertainty)
        pred_conf = -posterior.posterior_predictive().uncertainty()
        if pred_conf.dim() > 1:
            pred_conf = pred_conf.mean(tuple(range(1, pred_conf.dim())))
        self.pred_conf_pr.update(pred_conf, y)
        self.log(f"{self.logging_key}/predictive_confidence_auc_pr", self.pred_conf_pr)
        self.pred_conf_roc.update(pred_conf, y)
        self.log(f"{self.logging_key}/predictive_confidence_auc_roc", self.pred_conf_roc)


class NaturalPosteriorNetworkOodCorruptedLightningModule(pl.LightningModule):
    """
    Lightning module for training and evaluating NatPN.
    """

    def __init__(
        self,
        model: NaturalPosteriorNetworkModel,
        logging_key: str,
        entropy_weight: float,
    ):
        """
        Args:
            model: The model to train or evaluate. If training, this *must* be a
                :class:`NaturalPosteriorNetworkModel`.
        """
        super().__init__()
        self.model = model
        self.logging_key = logging_key
        self.accuracy = Accuracy(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.brier_score = BrierScore(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.loss = BayesianLoss(entropy_weight, 0)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        X, y_true = batch
        y_pred, log_prob = self.model.forward(X)
        loss, _, _ = self.loss.forward(y_pred, y_true)

        prefix = f"test/{self.logging_key}"
        self.log(f"{prefix}/loss", loss, prog_bar=True)
        self.log(f"{prefix}/log_prob", log_prob.mean())
        self._compute_metrics(prefix, y_pred, y_true)

    def _compute_metrics(self, prefix: str, y_pred: D.Posterior, y_true: torch.Tensor) -> None:
        self.accuracy.update(y_pred.maximum_a_posteriori().mean(), y_true)
        self.log(f"{prefix}/accuracy", self.accuracy, prog_bar=True)
        probs = y_pred.maximum_a_posteriori().expected_sufficient_statistics()
        self.brier_score.update(probs, y_true)
        self.log(f"{prefix}/brier_score", self.brier_score)
        