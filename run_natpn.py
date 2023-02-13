import pytorch_lightning as pl
from src.datasets import DATASET_REGISTRY
from src.models import suppress_pytorch_lightning_logs
from src.models.natpn.lightning import NaturalPosteriorNetwork

seed = 42
reconst_weights = [0]

for reconst_weight in reconst_weights:
    suppress_pytorch_lightning_logs()
    pl.seed_everything(seed)
    dm = DATASET_REGISTRY["blob"](seed=seed)
    dm.prepare_data()
    dm.setup("test")

    trainer_params = dict(
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        max_epochs=50,
        gpus=0,
    )

    params_dict = dict(
        latent_dim=4,
        encoder="tabular",
        encoder_act="relu",
        flow="radial",
        flow_num_layers=8,
        coeff=1,
        n_power_iterations=1,
        bn_out=True,
        learning_rate=1e-1,
        learning_rate_nf=1e-2,
        reconst_weight=reconst_weight,
        optim="adamw",
        warmup_epochs=1,
        trainer_params=trainer_params,
    )

    estimator = NaturalPosteriorNetwork(**params_dict)
    estimator.fit(dm)
    result_id = estimator.score(dm)
    result_ood = estimator.score_ood_detection(dm)

    print(result_id)
    print(result_ood)
