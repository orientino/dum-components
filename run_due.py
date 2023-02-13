import pytorch_lightning as pl
from src.datasets import DATASET_REGISTRY
from src.models import suppress_pytorch_lightning_logs
from src.models.dkl.nn.due import DeepKernelLearning

suppress_pytorch_lightning_logs()
seeds = [42]

for seed in seeds:
    pl.seed_everything(seed)
    dm = DATASET_REGISTRY["blob"](seed=seed)
    dm.prepare_data()
    dm.setup("test")

    params_dict = dict(
        latent_dim=4,
        encoder="tabular",
        encoder_act="relu",
        kernel="RBF",
        residual=True,
        spectral=(False, False, False),
        coeff=1,
        n_power_iterations=1,
        
        # Training parameters
        learning_rate=1e-3,
        learning_rate_gp=1e-2,
        early_stopping=False,
        optim="adamw",
        scheduler="cosine5e-4",
        max_epochs=50,
        gpus=0,
    )
    
    estimator = DeepKernelLearning(**params_dict)
    estimator.fit(dm)
    result_id = estimator.score(dm)
    result_ood = estimator.score_ood_detection(dm)

    print(result_id)
    print(result_ood)
