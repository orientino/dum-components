import torch
import torch.nn as nn
import logging
from typing import Tuple

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn import cluster
import torchvision.models as models
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from src.models.dkl.lightning.lightning_module import DeepKernelLearningLightningModule
from src.models.dkl.lightning.lightning_module_gp import DeepKernelLearningGPLightningModule
from src.models.dkl.lightning.lightning_module_ood import DeepKernelLearningOodTestingLightningModule, DeepKernelLearningOodCorruptedLightningModule
from src.architectures.encoder import *
from src.datasets import DataModule

logger = logging.getLogger(__name__)


class DeepKernelLearning(gpytorch.Module):
    def __init__(
        self, 
        latent_dim: int = 16,
        encoder: EncoderType = "resnet18", 
        kernel: str = "RBF",
        n_inducing_points: int = 0,
        n_inducing_points_scaler: float = 1.0,
        dropout: int = 0,
        residual: bool = True,
        spectral: Tuple[bool] = (False, False, False),
        lipschitz_constant: float = 1.0, 
        n_power_iterations: int = 1,
        bn_out: bool = False,
        reconst_weight: float = 0.0,
        pretrained_enc_path: str = "",

        # Training parameters
        learning_rate: float = 1e-3,
        learning_rate_head: float = 1e-3,
        early_stopping: bool = False,
        optim: str = "sgd",
        scheduler: str = "cosine5e-4",
        training_schema: str = "joint-all",
        max_epochs: int = 100, 
        finetune_epochs: int = 0,
        gpus: int = 0,
        logger = None,
    ):
        """
        Args:
            latent_dim: The dimension of the latent space that the encoder should map to.
            encoder: The type of encoder to use which maps the input to the latent space.
            kernel: The type of kernel function used in the Gaussian Process.
            n_inducing_points: The number of inducing points used (default = #classes)
            n_inducing_points_scaler: The scaler for the n_inducing points 
            dropout: The dropout probability to use for dropout layers in the encoder.
            residual: Used only for the TabularEncoder
            spectral: The usage of spectral normalization for different layers
                (spectral_fc, spectral_conv, spectral_bn)
            lipschitz_constant: The spectral normalization's Lipschitz constant parameter.
            n_power_iterations: The spectral normalization's power iteration parameter.
            bn_out: Append the batch normalization layer at the end of the encoder.
            reconst_weight: The reconstruction term coefficient in the loss.
            pretrained_enc_path: The path used to load a pretrained encoder.
            learning_rate: The learning rate to use for training encoder.
            learning_rate_head: The learning rate to use for training the Gaussian Process.
            early_stopping: Early stopping for the training.
            optim: The optimizer used.
            scheduler: The scheduler used to schedule the optimizer.
            training_schema: The training schema joint or frozen.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.kernel = kernel
        self.n_inducing_points = n_inducing_points
        self.n_inducing_points_scaler = n_inducing_points_scaler
        self.dropout = dropout
        self.residual = residual
        self.spectral = spectral
        self.lipschitz_constant = lipschitz_constant
        self.n_power_iterations = n_power_iterations
        self.bn_out = bn_out
        self.reconst_weight = reconst_weight
        self.pretrained_enc_path = pretrained_enc_path
        self.training_schema = training_schema
        self.max_epochs = max_epochs
        self.finetune_epochs = finetune_epochs

        self.lightning_params = dict(
            learning_rate=learning_rate,
            learning_rate_head=learning_rate_head,
            early_stopping=early_stopping,
            optim=optim,
            scheduler=scheduler,
        )
        self.trainer_params = dict(
            gpus=gpus,
            logger=logger,
            enable_model_summary=True,
        )

    def forward(self, x: torch.Tensor):
        """ 
        Return the reconstructed input only if the reconstruction weight > 0 and during training phase
        """
        if self.reconst_weight > 0 and self.encoder.training:
            z, r = self.encoder(x)
            y_hat = self.decoder(r)
            return self.gp(z), y_hat
        else:
            z = self.encoder(x)
            return self.gp(z)

    def fit(self, data: DataModule):
        self._init_model(data.input_size, data.num_classes, data.train_dataset)
        self._fit_model(data)

    def score(self, data: DataModule):
        logger.info("Evaluating on test set...")
        module = DeepKernelLearningLightningModule(self)
        trainer = Trainer(**self.trainer_params)
        out = trainer.test(module, data, verbose=False)

        results = {}
        for o in out:
            dataset = next(iter(o.keys())).split("/")[1]
            results[dataset] = {k.split("/")[2]: v for k, v in o.items()}

        return results

    def score_ood_detection(self, data: DataModule):
        results = {}
        trainer = Trainer(**self.trainer_params)
        for dataset, loader in data.ood_dataloaders().items():
            logger.info("Evaluating in-distribution vs. %s...", dataset)
            module = DeepKernelLearningOodTestingLightningModule(
                self, logging_key=f"ood/{dataset}"
            )
            result = trainer.test(module, loader, verbose=False)
            results[dataset] = {k.split("/")[2]: v for k, v in result[0].items()}

        return results

    def score_ood_corrupted(self, data: DataModule):
        logger.info("Evaluating on corrupted test set...")
        results = {"loss": [], "accuracy": [], "brier": []}
        trainer = Trainer(**self.trainer_params)
        for i, loader in enumerate(data.ood_corrupted_dataloaders()):
            module = DeepKernelLearningOodCorruptedLightningModule(
                self, logging_key=f"ood1")
            out = trainer.test(module, loader, verbose=False)
            for o in out:
                o = {k.split("/")[2]: v for k, v in o.items()}
                results["loss"].append(o["loss"])
                results["accuracy"].append(o["accuracy"])
                results["brier"].append(o["brier_score"])

        return results

    def load(self, path: str, data: DataModule):
        self._init_model(data.input_size, data.num_classes, data.train_dataset)
        self.load_state_dict(torch.load(path))

    def _fit_model(self, data):
        logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [lr_monitor] if self.trainer_params["logger"] else []

        # Select joint training schema
        # joint-all:                Train GP + encoder
        # joint-frozen-enc:         Train GP
        # reset-joint-all:          Train GP + encoder (reset last layer)
        # reset-joint-frozen-enc:   Train GP + encoder (reset last layer only)
        if self.training_schema == "joint-all":
            logger.info("Running joint-all...")
        elif self.training_schema == "joint-frozen-enc":
            logger.info("Running joint-frozen-enc...")
            self.encoder.requires_grad_(False)
            if self.bn_out:
                self.encoder.bn_out.requires_grad_(True)
        elif self.training_schema == "reset-joint-all":
            logger.info("Reset encoder's last linear layer. Running joint-all...")
            self.encoder.linear_latent.reset_parameters()
        elif self.training_schema == "reset-joint-frozen-enc":
            logger.info("Reset encoder's last linear layer. Running joint-frozen-enc (only enc's body is frozen)...")
            self.encoder.linear_latent.reset_parameters()
            self.encoder.requires_grad_(False)
            self.encoder.linear_latent.requires_grad_(True)
            if self.bn_out:
                self.encoder.bn_out.requires_grad_(True)
        else:
            raise NotImplementedError

        # Run training
        trainer = Trainer(
            callbacks=callbacks,
            max_epochs=self.max_epochs,
            **self.trainer_params,
        )
        if self.training_schema != "joint-frozen-enc":
            train_module = DeepKernelLearningLightningModule(
                self,
                max_epochs=self.max_epochs,
                n_batches=len(data.train_dataloader()),
                **self.lightning_params,
            )
        else:
            train_module = DeepKernelLearningGPLightningModule(
                self,
                max_epochs=self.max_epochs,
                n_batches=len(data.train_dataloader()),
                **self.lightning_params,
            )
        trainer.fit(train_module, data)

        # Run fine-tuning
        if self.finetune_epochs > 0:
            logger.info("Running fine-tuning...")
            self.encoder.requires_grad_(False)
            trainer = Trainer(
                callbacks=callbacks,
                max_epochs=self.finetune_epochs,
                **self.trainer_params,
            )
            finetune_module = DeepKernelLearningGPLightningModule(
                self,
                max_epochs=self.finetune_epochs,
                n_batches=len(data.train_dataloader()),
                phase="finetune",
                **self.lightning_params,
            )
            trainer.fit(finetune_module, data)


    def _init_model(self, input_size, num_classes, train_dataset):
        # Initialize encoder and optionally the decoder
        encoder_type = self.encoder
        if self.encoder == "tabular":
            assert len(input_size) == 1, "Tabular encoder only allows for one-dimensional inputs."
            self.encoder = TabularEncoder(
                input_size[0], 
                [64] * 3, 
                self.latent_dim,
                spectral=self.spectral,
                residual=self.residual,
                lipschitz_constant=self.lipschitz_constant,
                n_power_iterations=self.n_power_iterations,
                reconst_out=self.reconst_weight>0,
            )
            if self.reconst_weight > 0:
                self.decoder = TabularEncoder(
                    self.latent_dim,
                    [64] * 3,
                    input_size[0],
                    spectral=self.spectral,
                    residual=self.residual,
                    lipschitz_constant=self.lipschitz_constant,
                    n_power_iterations=self.n_power_iterations,
                )
        elif self.encoder == "resnet18":
            self.encoder = ResNet(
                self.latent_dim, 
                num_classes,
                input_size[1], 
                residual=self.residual,
                spectral=self.spectral, 
                lipschitz_constant=self.lipschitz_constant,
                n_power_iterations=self.n_power_iterations,
                reconst_out=self.reconst_weight>0,
            )
            if self.reconst_weight > 0:
                self.decoder = ResNetDecoder(
                    spectral=self.spectral,
                    lipschitz_constant=self.lipschitz_constant,
                    n_power_iterations=self.n_power_iterations,
                )
        elif self.encoder == "wide-resnet":
            self.encoder = WideResNet(
                self.latent_dim,
                num_classes,
                input_size[1],
                spectral=self.spectral,
                dropout_rate=self.dropout
            )
        elif self.encoder == "resnet18-tv":
            self.encoder = tv_resnet18(True, self.latent_dim, num_classes)
        elif self.encoder == "resnet50":
            self.encoder = tv_resnet50(True, self.latent_dim, num_classes)
        elif self.encoder == "efficientnet_v2_s":
            self.encoder = tv_efficientnet_v2_s(True, self.latent_dim, num_classes)
        elif self.encoder == "swin_t":
            self.encoder = tv_swin_t(True, self.latent_dim, num_classes)
        else:
            raise NotImplementedError

        if self.pretrained_enc_path != "":
            logger.info(f"Loading the pretrained weights from: {self.pretrained_enc_path}")
            self.encoder.load_state_dict(torch.load(self.pretrained_enc_path))
            if self.reconst_weight > 0:
                pretrained_decoder_file = self.pretrained_enc_path.split("/")[-1].replace(".pt", "_decoder.pt")
                pretrained_decoder_path = "/".join(self.pretrained_enc_path.split("/")[:-1] + [pretrained_decoder_file])
                logger.info(f"Loading the pretrained decoder weights from: {pretrained_decoder_path}")
                self.decoder.load_state_dict(torch.load(pretrained_decoder_path))

        self.encoder.linear = nn.Identity()
        if self.bn_out:
            if encoder_type in ["resnet18", "wide-resnet"]:
                self.encoder.add_bn_output(self.latent_dim)
            else:
                self.encoder.bn_out = nn.BatchNorm1d(self.latent_dim)

        # Initialize GP
        if self.n_inducing_points == 0:
            self.n_inducing_points = int(num_classes * self.n_inducing_points_scaler)
        initial_inducing_points, initial_lengthscale = self._initial_values(
            train_dataset, self.encoder, self.n_inducing_points
        )
        self.gp = GP(
            num_outputs=num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=self.kernel,
        )

        # Initialize DUE's loss
        self.likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
        self.loss = lambda x, y: -VariationalELBO(self.likelihood, self.gp, num_data=len(train_dataset))(x, y)

    def _initial_values(self, train_dataset, encoder, n_inducing_points):
        steps = 10
        idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
        f_X_samples = []

        with torch.no_grad():
            for i in range(steps):
                X_sample = torch.stack([train_dataset[j][0] for j in idx[i]])
                f_X_sample = encoder(X_sample)
                f_X_sample = f_X_sample[0] if self.reconst_weight > 0 else f_X_sample
                f_X_samples.append(f_X_sample)
            f_X_samples = torch.cat(f_X_samples)

        initial_inducing_points = self._get_initial_inducing_points(
            f_X_samples.numpy(), n_inducing_points
        )
        initial_lengthscale = self._get_initial_lengthscale(f_X_samples)
        return initial_inducing_points, initial_lengthscale

    def _get_initial_inducing_points(self, f_X_sample, n_inducing_points):
        kmeans = cluster.MiniBatchKMeans(
            n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
        )
        kmeans.fit(f_X_sample)
        initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)
        return initial_inducing_points

    def _get_initial_lengthscale(self, f_X_samples):
        initial_lengthscale = torch.pdist(f_X_samples).mean()
        return initial_lengthscale


class GP(ApproximateGP):
    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        initial_inducing_points,
        kernel="RBF",
    ):
        n_inducing_points = initial_inducing_points.shape[0]
        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])
        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
            )

        super().__init__(variational_strategy)

        kwargs = {"batch_shape": batch_shape}
        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param
