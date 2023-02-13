import os
import random
import time
import tempfile
import logging
from typing import List, Tuple, cast
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import seml
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import Accuracy, MeanSquaredError, AUROC
from torchmetrics.functional import accuracy
from sacred import Experiment
from src.architectures.activation import ActivationType
from src.architectures.encoder import *
from src.metrics import BrierScore
from src.datasets import DATASET_REGISTRY
from src.models import suppress_pytorch_lightning_logs

ex = Experiment()
seml.setup_logger(ex)
logger = logging.getLogger(__name__)


class LitModel(pl.LightningModule):
    def __init__(
        self, 
        model: EncoderType = "resnet18",
        latent_dim: int = 100,
        act_fun: ActivationType = "relu",
        dropout: float = 0.0,
        residual: bool = True,
        spectral: Tuple[bool] = (False, False, False),
        coeff: float = 1.0,
        n_power_iterations: int = 1,
        reconst_weight: float = 0.0,

        n_classes: int = 10,
        input_size: int = 32,

        pretrained_path: str = "",
        learning_rate: float = 0.05, 
        weight_decay: float = 1.0, 
        n_batches: int = 10,
        optimizer: str = "sgd",
        scheduler: str = "cosine",
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model == "tabular":
            self.model = TabularEncoder(
                2, 
                [64] * 3, 
                n_classes,
                act=act_fun,
                spectral=spectral,
                residual=residual,
                coeff=coeff,
                n_power_iterations=n_power_iterations,
                reconst_out=reconst_weight>0,
            )
        elif model == "resnet18":
            self.model = ResNet(
                latent_dim,
                n_classes, 
                input_size, 
                act=act_fun,
                residual=residual,
                spectral=spectral,
                coeff=coeff,
                n_power_iterations=n_power_iterations,
                reconst_out=reconst_weight>0,
            )
            if reconst_weight > 0:
                self.decoder = ResNetDecoder(
                    act=act_fun,
                    spectral=spectral,
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )
        elif model == "wide-resnet":
            self.model = WideResNet(
                latent_dim,
                n_classes,
                input_size, 
                spectral=spectral,
                dropout_rate=dropout,
                residual=residual,
            )
        elif model == "resnet50":
            self.model = tv_resnet50(True, latent_dim, n_classes)
        elif model == "efficientnet_v2_s":
            self.model = tv_efficientnet_v2_s(True, latent_dim, n_classes)
        elif model == "swin_t":
            self.model = tv_swin_t(True, latent_dim, n_classes)
        else:
            raise NotImplementedError

        if pretrained_path != "":
            self.model.load_state_dict(torch.load(pretrained_path))
            if reconst_weight > 0 and model == "resnet18":
                pretrained_decoder_file = pretrained_path.split("/")[-1].replace(".pt", "_decoder.pt")
                pretrained_decoder_path = "/".join(pretrained_path.split("/")[:-1] + [pretrained_decoder_file])
                self.decoder.load_state_dict(torch.load(pretrained_decoder_path))

        self.data2dist = {0: "id", 1: "ood", 2: "oodom"}

    def forward(self, X):
        if self.hparams.reconst_weight > 0 and self.training:
            out, rec = self.model(X)
            logits = F.log_softmax(out, dim=1)
            rec_loss = F.mse_loss(X, self.decoder(rec))
            return logits, rec_loss
        else: 
            out = self.model(X)
            logits = F.log_softmax(out, dim=1)
            return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        if self.hparams.reconst_weight > 0:
            logits, rec_loss = self(X)
        else:
            logits = self(X)
            rec_loss = 0

        loss = F.nll_loss(logits, y) + self.hparams.reconst_weight * rec_loss
        self.log("train/loss", loss)
        self.log("train/rec_loss", rec_loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self._evaluate(batch, dataloader_idx, f"val/{self.data2dist[dataloader_idx]}")

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self._evaluate(batch, dataloader_idx, f"test/{self.data2dist[dataloader_idx]}")

    def _evaluate(self, batch, dataloader_idx, stage=None):
        X, y = batch
        logits = self(X)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True, add_dataloader_idx=False)
            self.log(f"{stage}/acc", acc, prog_bar=True, add_dataloader_idx=False)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            raise NotImplementedError

        # Initialize scheduler
        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.2,
                    patience=self.trainer.max_epochs // 5,
                    min_lr=1e-7,
                ),
                "monitor": "val/id/loss",
            }
        elif self.hparams.scheduler == "multistep":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[60, 120, 160], gamma=0.2
                )
            }
        elif self.hparams.scheduler == "multistep10":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[10, 20], gamma=0.2
                )
            }
        elif self.hparams.scheduler == "multistep100":
           lr_scheduler = {
               "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                   optimizer, milestones=[100, 150, 175], gamma=0.2
               )
           }
        elif self.hparams.scheduler == "cosine5e-4":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.max_epochs * self.hparams.n_batches,
                    eta_min=5e-4,
                ),
                "interval": "step",
            }
        elif self.hparams.scheduler == "cosine1e-5":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.max_epochs * self.hparams.n_batches,
                    eta_min=1e-5,
                ),
                "interval": "step",
            }
        elif self.hparams.scheduler == "cosine5e-5":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.max_epochs * self.hparams.n_batches,
                    eta_min=5e-5,
                ),
                "interval": "step",
            }
        else:
            raise NotImplementedError

        return [optimizer], lr_scheduler


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
            db_collection, overwrite=overwrite))


@ex.automain
def run(
        project_name: str,
        dataset_name: str,
        seed: int,

        # Model parameters
        *,
        directory_model: str,
        model: EncoderType,
        latent_dim: int,
        act_fun: ActivationType = "relu",
        dropout: float = 0.0,
        residual: bool = True,
        spectral: Tuple[bool] = (False, False, False),
        coeff: float = 1.0,
        n_power_iterations: int = 1,
        reconst_weight: float = 0.0,

        # Training parameters
        pretrained_path: str = "",
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        optimizer: str = "adamw",
        scheduler: str = "multistep",
        max_epochs: int = 1,
        gpus: int = 0,
):
    suppress_pytorch_lightning_logs()
    pl.seed_everything(seed)

    # ---------------------------------------------------------------------------------------------
    # LOAD THE DATASET

    dm = DATASET_REGISTRY[dataset_name](seed=seed)
    dm.prepare_data()
    dm.setup("test")

    # ---------------------------------------------------------------------------------------------
    # CREATE THE MODEL

    while True:
        random_name = str(random.randint(0, 1e7))
        model_path = Path(directory_model) / f"{project_name}-{random_name}"
        if not os.path.exists(model_path):
            break
    os.makedirs(model_path)
    
    wandb_logger = WandbLogger(save_dir=model_path,
                                name=f"{project_name}-{random_name}",
                                project=project_name,
                                entity="tum-thesis")

    params_dict = dict(
        model=model,
        latent_dim=latent_dim,
        act_fun=act_fun,
        dropout=dropout,
        residual=residual,
        spectral=spectral,
        coeff=coeff,
        n_power_iterations=n_power_iterations,
        reconst_weight=reconst_weight,

        n_classes=dm.num_classes,
        input_size=dm.input_size[1],

        pretrained_path=pretrained_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_batches=len(dm.train_dataloader()),
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
    )

    model = LitModel(**params_dict)

    params_dict.update(dict(seed=seed, dataset_name=dataset_name))
    wandb_logger.experiment.config.update(params_dict)

    # ---------------------------------------------------------------------------------------------
    # TRAINING

    t0 = time.time()

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # EarlyStopping(monitor="val/loss", mode="min", min_delta=1e-3, patience=40),
    ]

    if dataset_name == "camelyon":
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer_checkpoint = ModelCheckpoint(Path(tmp_dir) / "training", monitor="val/ood/acc", mode="max")
            callbacks.append(trainer_checkpoint)

    trainer = pl.Trainer(
        max_epochs=params_dict["max_epochs"],
        gpus=gpus,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, dm)

    if dataset_name == "camelyon":
        logger.info(f"Loading checkpoint from {trainer_checkpoint.best_model_path}")
        model = LitModel.load_from_checkpoint(trainer_checkpoint.best_model_path)

    t1 = time.time()

    # ---------------------------------------------------------------------------------------------
    # TESTING

    results = trainer.test(model, dm.test_dataloader())

    # ---------------------------------------------------------------------------------------------
    # SAVING & RESULTS

    torch.save(model.model.state_dict(), model_path / f"{random_name}.pt")
    if reconst_weight > 0:
        torch.save(model.decoder.state_dict(), model_path / f"{random_name}_decoder.pt")

    fail_trace = {
        "model_path": model_path,
        # "fail_trace": seml.evaluation.get_results,
        "training_time": t1 - t0,
    }

    return {**results[0], **fail_trace}
