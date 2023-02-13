from __future__ import annotations
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Tuple, cast, Dict, List, Optional

import torch
import torch.nn as nn
from lightkit import BaseEstimator
from pytorch_lightning.callbacks import LearningRateMonitor
from src.datasets import DataModule, OutputType
from src.architectures.encoder import *
from src.architectures.activation import *
from src.models.natpn.nn import *
from src.models.natpn.nn.flow import *
from src.models.natpn.nn.output import CategoricalOutput, NormalOutput, PoissonOutput
from .lightning_module import NaturalPosteriorNetworkLightningModule
from .lightning_module_flow import NaturalPosteriorNetworkFlowLightningModule
from .lightning_module_ood import NaturalPosteriorNetworkOodTestingLightningModule, NaturalPosteriorNetworkOodCorruptedLightningModule

logger = logging.getLogger(__name__)


class NaturalPosteriorNetwork(BaseEstimator):
    """
    Estimator for the Natural Posterior Network
    """

    #: The fitted model.
    model_: NaturalPosteriorNetworkModel
    #: The input size of the model.
    input_size_: torch.Size
    #: The output type of the model.
    output_type_: OutputType
    #: The number of classes the model predicts if ``output_type_ == "categorical"``.
    num_classes_: Optional[int]

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        encoder: EncoderType = "tabular",
        encoder_act: ActivationType = "relu",
        flow: FlowType = "radial",
        flow_num_layers: int = 8,
        prior_coeff: float = 1.0,
        budget: CertaintyBudget = "normal",
        dropout: float = 0.0,
        residual: bool = True,
        spectral: Tuple[bool] = (False, False, False),
        coeff: float = 1.0, 
        n_power_iterations: int = 1,
        bn_out: bool = False,
        entropy_weight: float = 1e-5,
        reconst_weight: float = 0.0,
        pretrained_enc_path: str = "",

        # Training parameters
        learning_rate: float = 1e-3,
        learning_rate_nf: float = 1e-4,
        early_stopping: bool = False,
        warmup_epochs: int = 5,
        finetune_epochs: int = 0,
        optim: str = "adam",
        scheduler: str = "multistep",
        training_schema: str = "joint-all",
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            latent_dim: The dimension of the latent space that the encoder should map to.
            encoder: The type of encoder to use which maps the input to the latent space.
            encoder_act: The encoder's activation function used
            flow: The type of flow which produces log-probabilities from the latent
                representations.
            flow_num_layers: The number of layers to use for the flow. If ``flow`` is set to
                ``"maf"``, this sets the number of masked autoregressive layers. In between each
                of these layers, another batch normalization layer is added.
            prior_coeff: prior scaler for the `evidence parameter n`.
            budget: The certainty budget to use to scale the log-probabilities produced
                by the normalizing flow.
            dropout: The dropout probability to use for dropout layers in the encoder.
            residual: Used only for the TabularEncoder
            spectral: The usage of spectral normalization for different layers
                (spectral_fc, spectral_conv, spectral_bn)
            coeff: The spectral normalization's Lipschitz constant parameter.
            n_power_iterations: The spectral normalization's power iteration parameter.
            bn_out: Append the batch normalization layer at the end of the encoder.
            learning_rate: The learning rate to use for training encoder, flow, and linear output
                layer. Applies to warm-up, actual training, and fine-tuning.
            early_stopping: Early stopping for the training.
            entropy_weight: The strength of the entropy regularizer for the Bayesian loss used for
                the main training procedure.
            reconst_weight: The reconstruction term coefficient in the loss.
            warmup_epochs: The number of epochs to run warm-up for. Should be used if the latent
                space is high-dimensional and/or the normalizing flow is complex, i.e. consists of
                many layers.
            finetune_epochs: The number of epochs to run fine-tuning after the main training 
                loop for. It helps to improve out-of-distribution detection.
            trainer_params: Additional parameters which are passed to the PyTorch Ligthning
                trainer. These parameters apply to all fitting runs as well as testing.
        """
        super().__init__(
            user_params=trainer_params,
            overwrite_params=dict(
                # log_every_n_steps=1,
                check_val_every_n_epoch=1,
                enable_progress_bar=True,
                enable_model_summary=True,
            ),
        )

        self.latent_dim = latent_dim
        self.encoder = encoder
        self.encoder_act = encoder_act
        self.flow = flow
        self.flow_num_layers = flow_num_layers
        self.prior_coeff = prior_coeff
        self.budget = budget
        self.dropout = dropout
        self.residual = residual
        self.spectral = spectral
        self.coeff = coeff
        self.n_power_iterations = n_power_iterations
        self.bn_out = bn_out
        self.entropy_weight = entropy_weight
        self.reconst_weight = reconst_weight
        self.pretrained_enc_path = pretrained_enc_path
        self.learning_rate = learning_rate
        self.learning_rate_nf = learning_rate_nf
        self.early_stopping = early_stopping
        self.warmup_epochs = warmup_epochs
        self.finetune_epochs = finetune_epochs
        self.optim = optim
        self.scheduler = scheduler
        self.training_schema = training_schema

        assert len(spectral) == 3, "'spectral' requires the input as (spectral_fc, spectral_conv, spectral_bn)"

    # ---------------------------------------------------------------------------------------------
    # RUNNING THE MODEL

    def fit(self, data: DataModule) -> NaturalPosteriorNetwork:
        """
        Fits the Natural Posterior Network with the provided data. Fitting sequentially runs
        warm-up (if ``self.warmup_epochs > 0``), the main training loop, and fine-tuning (if
        ``self.finetune_epochs > 0``).

        Args:
            data: The data to fit the model with.

        Returns:
            The estimator whose ``model_`` property is set.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = self._init_model(
                data.output_type,
                data.input_size,
                data.num_classes if data.output_type == "categorical" else 0,
            )
            self.model_ = self._fit_model(model, data, Path(tmp_dir))

        # Assign additional fitted attributes
        self.input_size_ = data.input_size
        self.output_type_ = data.output_type
        try:
            self.num_classes_ = data.num_classes
        except NotImplementedError:
            self.num_classes_ = None

        return self

    def score(self, data: DataModule) -> Dict[str, float]:
        """
        Measures the model performance on the given data.

        Args:
            data: The data for which to measure the model performance.

        Returns:
            A dictionary mapping metrics to their values. This dictionary includes a measure of
            accuracy (`"accuracy"` for classification and `"rmse"` for regression) and a
            calibration measure (`"brier_score"` for classification and `"calibration"` for
            regression).
        """
        logger.info("Evaluating on test set...")
        module = NaturalPosteriorNetworkLightningModule(self.model_)
        out = self.trainer().test(module, data, verbose=False)

        results = {}
        for o in out:
            dataset = next(iter(o.keys())).split("/")[1]
            results[dataset] = {k.split("/")[2]: v for k, v in o.items()}

        return results

    def score_ood_detection(self, data: DataModule) -> Dict[str, Dict[str, float]]:
        """
        Measures the model's ability to detect out-of-distribution data.

        Args:
            data: The data module which provides one or more datasets that contain test data along
                with out-of-distribution data.

        Returns:
            A nested dictionary which provides for multiple out-of-distribution datasets (first
            key) multiple metrics for measuring epistemic and aleatoric uncertainty.
        """
        results = {}
        for dataset, loader in data.ood_dataloaders().items():
            logger.info("Evaluating in-distribution vs. %s...", dataset)
            module = NaturalPosteriorNetworkOodTestingLightningModule(
                self.model_, logging_key=f"ood/{dataset}"
            )
            result = self.trainer().test(module, loader, verbose=False)
            results[dataset] = {k.split("/")[2]: v for k, v in result[0].items()}

        return results

    def score_ood_corrupted(self, data: DataModule):
        logger.info("Evaluating on corrupted test set...")
        results = {"loss": [], "accuracy": [], "brier": [], "log_prob": []}
        for i, loader in enumerate(data.ood_corrupted_dataloaders()):
            module = NaturalPosteriorNetworkOodCorruptedLightningModule(
                self.model_, logging_key=f"ood1", entropy_weight=self.entropy_weight)
            out = self.trainer().test(module, loader, verbose=False)
            for o in out:
                o = {k.split("/")[2]: v for k, v in o.items()}
                results["loss"].append(o["loss"])
                results["accuracy"].append(o["accuracy"])
                results["brier"].append(o["brier_score"])
                results["log_prob"].append(o["log_prob"])

        return results

    # ---------------------------------------------------------------------------------------------
    # PERSISTENCE

    @property
    def persistent_attributes(self) -> List[str]:
        return [k for k in self.__annotations__ if k != "model_"]

    def save_parameters(self, path: Path) -> None:
        params = {
            k: (
                v
                if k != "trainer_params"
                else {kk: vv for kk, vv in cast(Dict[str, Any], v).items() if kk != "logger"}
            )
            for k, v in self.get_params().items()
        }
        data = json.dumps(params, indent=4)
        with (path / "params.json").open("w+") as f:
            f.write(data)

    def save_attributes(self, path: Path) -> None:
        super().save_attributes(path)
        torch.save(self.model_.state_dict(), path / "parameters.pt")

    def load_attributes(self, path: Path) -> None:
        super().load_attributes(path)
        parameters = torch.load(path / "parameters.pt")
        model = self._init_model(self.output_type_, self.input_size_, self.num_classes_ or 0)
        model.load_state_dict(parameters)
        self.model_ = model

    # ---------------------------------------------------------------------------------------------
    # UTILS

    def _fit_model(
        self, 
        model: NaturalPosteriorNetworkModel, 
        data: DataModule, 
        tmp_dir: Path, 
    ) -> NaturalPosteriorNetworkModel:
        logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [lr_monitor] if self.trainer().logger else []

        # Select joint training schema
        # joint-all:                Pretrained -> Joint (all)
        # joint-frozen-enc:         Pretrained -> Joint (NF + output)
        # reset-joint-all:          Pretrained (reset linear) -> Joint (all)
        # reset-joint-frozen-enc:   Pretrained (reset linear) -> Joint (NF + output + last linear)
        if self.training_schema == "joint-all":
            logger.info("Running joint-all...")
        elif self.training_schema == "joint-frozen-enc":
            logger.info("Running joint-frozen-enc...")
            model.encoder.requires_grad_(False)
            if self.bn_out:
                model.encoder.bn_out.requires_grad_(True)
        elif self.training_schema == "reset-joint-all":
            logger.info("Reset encoder's last linear layer. Running joint-all...")
            model.encoder.linear_latent.reset_parameters()
        elif self.training_schema == "reset-joint-frozen-enc":
            logger.info("Reset encoder's last linear layer. Running joint-frozen-enc (only enc's body is frozen)...")
            model.encoder.linear_latent.reset_parameters()
            model.encoder.requires_grad_(False)
            model.encoder.linear_latent.requires_grad_(True)
            if self.bn_out:
                model.encoder.bn_out.requires_grad_(True)
        else:
            raise NotImplementedError

        # Run warmup the Normalizing Flow
        if self.warmup_epochs > 0:
            logger.info("Running warmup...")
            trainer = self.trainer(
                callbacks=callbacks,
                enable_checkpointing=False,
                max_epochs=self.warmup_epochs,
            )
            warmup_module = NaturalPosteriorNetworkFlowLightningModule(
                model, 
                learning_rate=self.learning_rate_nf, 
                early_stopping=False,
                phase="warmup",
            )
            trainer.fit(warmup_module, data)

        # Run joint training
        trainer = self.trainer(
            callbacks=callbacks,
        )
        train_module = NaturalPosteriorNetworkLightningModule(
            model,
            learning_rate=self.learning_rate,
            learning_rate_nf=self.learning_rate_nf,
            entropy_weight=self.entropy_weight,
            reconst_weight=self.reconst_weight,
            early_stopping=self.early_stopping,
            optim=self.optim,
            scheduler=self.scheduler,
            n_batches=len(data.train_dataloader()),
        )
        trainer.fit(train_module, data)
        best_module = train_module

        # Run fine-tuning
        if self.finetune_epochs > 0:
            logger.info("Running fine-tuning...")
            trainer = self.trainer(
                callbacks=callbacks,
                max_epochs=self.finetune_epochs
            )
            finetune_module = NaturalPosteriorNetworkFlowLightningModule(
                cast(NaturalPosteriorNetworkModel, best_module.model),
                learning_rate=self.learning_rate_nf,
                early_stopping=self.early_stopping,
                phase="finetune",
            )
            trainer.fit(finetune_module, data)
            best_module = finetune_module

        return cast(NaturalPosteriorNetworkModel, best_module.model)

    def _init_model(
        self, output_type: OutputType, input_size: torch.Size, num_classes: int
    ) -> NaturalPosteriorNetworkModel:
        # Initialize encoder and optionally the decoder
        if self.encoder == "tabular":
            assert len(input_size) == 1, "Tabular encoder only allows for one-dimensional inputs."
            encoder = TabularEncoder(
                input_size[0], 
                [64] * 3, 
                self.latent_dim,
                act=self.encoder_act,
                spectral=self.spectral,
                residual=self.residual,
                coeff=self.coeff,
                n_power_iterations=self.n_power_iterations,
                bn_out=self.bn_out,
                reconst_out=self.reconst_weight>0,
            )
            if self.reconst_weight > 0:
                decoder = TabularEncoder(
                    self.latent_dim,
                    [64] * 3,
                    input_size[0],
                    act=self.encoder_act,
                    spectral=self.spectral,
                    residual=self.residual,
                    coeff=self.coeff,
                    n_power_iterations=self.n_power_iterations,
                )
        elif self.encoder == "resnet18":
            encoder = ResNet(
                self.latent_dim, 
                num_classes,
                input_size[1], 
                act=self.encoder_act,
                residual=self.residual,
                spectral=self.spectral, 
                coeff=self.coeff,
                n_power_iterations=self.n_power_iterations,
                reconst_out=self.reconst_weight>0,
            )
            if self.reconst_weight > 0:
                decoder = ResNetDecoder(
                    act=self.encoder_act,
                    spectral=self.spectral,
                    coeff=self.coeff,
                    n_power_iterations=self.n_power_iterations,
                )
        elif self.encoder == "wide-resnet":
            encoder = WideResNet(
                self.latent_dim,
                num_classes,
                input_size[1],
                spectral=self.spectral,
                dropout_rate=self.dropout
            )
        elif self.encoder == "resnet18-tv":
            encoder = tv_resnet18(True, self.latent_dim, num_classes)
        elif self.encoder == "resnet50":
            encoder = tv_resnet50(True, self.latent_dim, num_classes)
        elif self.encoder == "efficientnet_v2_s":
            encoder = tv_efficientnet_v2_s(True, self.latent_dim, num_classes)
        elif self.encoder == "swin_t":
            encoder = tv_swin_t(True, self.latent_dim, num_classes)
        else:
            raise NotImplementedError

        if self.pretrained_enc_path != "":
            logger.info(f"Loading the pretrained weights from: {self.pretrained_enc_path}")
            encoder.load_state_dict(torch.load(self.pretrained_enc_path))
            if self.reconst_weight > 0:
                pretrained_decoder_file = self.pretrained_enc_path.split("/")[-1].replace(".pt", "_decoder.pt")
                pretrained_decoder_path = "/".join(self.pretrained_enc_path.split("/")[:-1] + [pretrained_decoder_file])
                logger.info(f"Loading the pretrained decoder weights from: {pretrained_decoder_path}")
                decoder.load_state_dict(torch.load(pretrained_decoder_path))

        encoder.linear = nn.Identity()
        if self.bn_out:
            if self.encoder in ["resnet18", "wide-resnet"]:
                encoder.add_bn_output(self.latent_dim)
            else:
                encoder.bn_out = nn.BatchNorm1d(self.latent_dim)

        # Initialize flow
        if self.flow == "radial":
            flows = nn.ModuleList([RadialFlow(self.latent_dim, self.flow_num_layers)])
        elif self.flow == "maf":
            flows = nn.ModuleList([MaskedAutoregressiveFlow(self.latent_dim, self.flow_num_layers)])
        elif self.flow == "residual":
            flows = nn.ModuleList([ResidualFlow(self.latent_dim, self.flow_num_layers)])
        elif self.flow == "nsf":
            flows = nn.ModuleList([NeuralSplineFlow(self.latent_dim, self.flow_num_layers)])
        elif self.flow == "nsf-r":
            flows = nn.ModuleList([NeuralSplineFlow(self.latent_dim, self.flow_num_layers, resampled=True)])
        else:
            raise NotImplementedError

        # Initialize output
        if output_type == "categorical":
            output = CategoricalOutput(self.latent_dim, num_classes, self.prior_coeff)
        elif output_type == "normal":
            output = NormalOutput(self.latent_dim)
        elif output_type == "poisson":
            output = PoissonOutput(self.latent_dim)
        else:
            raise NotImplementedError

        return NaturalPosteriorNetworkModel(
            self.latent_dim,
            encoder=encoder,
            decoder=decoder if self.reconst_weight > 0 else None,
            flows=flows,
            output=output,
            budget=self.budget,
        )
