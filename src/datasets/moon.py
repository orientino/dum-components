import logging
from typing import Any, cast, Dict, Optional

import torch
import pandas as pd
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from lightkit.utils import PathType
from lightkit.data import DataLoader

from ._registry import register
from ._base import DataModule, OutputType
from ._utils import tabular_ood_dataset

logger = logging.getLogger(__name__)


@register("moon")
class MoonDataModule(DataModule):
    """
    Data module for the Moon toy dataset.
    """

    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None, n_samples: int = 1000):
        super().__init__(root, seed)
        self.n_samples = n_samples
        self.batch_size = 64
        self.did_setup = False
        self.did_setup_ood = False

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([2])

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
        logger.info("Preparing 'Moon Dataset'...")
        data_file_train = self.root / "moon" / "train.csv"
        data_file_test = self.root / "moon" / "test.csv"
        data_file_ood_val = self.root / "moon" / "ood_val.csv"
        data_file_ood_test = self.root / "moon" / "ood_test.csv"

        if not data_file_train.exists():
            logger.info("'Moon' could not be found locally. Creating new dataset...")
            data_file_train.parent.mkdir(parents=True, exist_ok=True)

            train = make_moons(n_samples=800, noise=0.1, random_state=0)
            train = pd.DataFrame(columns=[train[0][:, 0], train[0][:, 1], train[1]]).transpose()
            train.to_csv(data_file_train, sep=",")

            test = make_moons(n_samples=200, noise=0.1, random_state=0)
            test = pd.DataFrame(columns=[test[0][:, 0], test[0][:, 1], test[1]]).transpose()
            test.to_csv(data_file_test, sep=",")

            ood_val = make_blobs(n_samples=200, centers=[[-1, 2]], cluster_std=0.1)
            ood_val = pd.DataFrame(columns=[ood_val[0][:, 0], ood_val[0][:, 1], ood_val[1]+2]).transpose()
            ood_val.to_csv(data_file_ood_val, sep=",")

            ood_test = make_blobs(n_samples=200, centers=[[2, 2]], cluster_std=0.1)
            ood_test = pd.DataFrame(columns=[ood_test[0][:, 0], ood_test[0][:, 1], ood_test[1]+2]).transpose()
            ood_test.to_csv(data_file_ood_test, sep=",")

    def setup(self, stage: Optional[str] = None):
        if not self.did_setup:
            # Train, val, test dataset
            train_file = self.root / "moon" / "train.csv"
            test_file = self.root / "moon" / "test.csv"
            df = cast(pd.DataFrame, pd.read_csv(train_file, sep=","))
            df_test = cast(pd.DataFrame, pd.read_csv(test_file, sep=","))
            train, val = train_test_split(df, test_size=0.25, stratify=df.iloc[:, 2])

            X_train = torch.from_numpy(train.to_numpy()[:, :-1]).float()
            y_train = torch.from_numpy(train.to_numpy()[:, -1]).long()
            X_val = torch.from_numpy(val.to_numpy()[:, :-1]).float()
            y_val = torch.from_numpy(val.to_numpy()[:, -1]).long()
            X_test = torch.from_numpy(df_test.to_numpy()[:, :-1]).float()
            y_test = torch.from_numpy(df_test.to_numpy()[:, -1]).long()

            # OOD val dataset
            ood_val_file = self.root / "moon" / "ood_val.csv"
            ood_val = cast(pd.DataFrame, pd.read_csv(ood_val_file, sep=","))
            X_ood_val = torch.from_numpy(ood_val.to_numpy()[:, :-1]).float()

            # Initialize datasts
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)
            self.test_dataset = TensorDataset(X_test, y_test)
            # self.ood_val_dataset = TensorDataset(X_ood_val, y_ood_val)
            self.ood_val_dataset = tabular_ood_dataset(X_val, X_ood_val)
            self.did_setup = True
        
        if stage == "test" and not self.did_setup_ood:
            ood_test_file = self.root / "moon" / "ood_test.csv"
            ood_test = cast(pd.DataFrame, pd.read_csv(ood_test_file, sep=","))
            X_ood = torch.from_numpy(ood_test.to_numpy()[:, :-1]).float()

            self.ood_test_datasets = {
                "blob": tabular_ood_dataset(X_test, X_ood)
            }
            self.did_setup_ood = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return [
            DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2),
            DataLoader(self.ood_val_dataset, batch_size=self.batch_size, num_workers=2),
        ]

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)

    def ood_dataloaders(self):
        return {
            name: DataLoader(dataset, batch_size=self.batch_size, num_workers=2)
            for name, dataset in self.ood_test_datasets.items()
        }
