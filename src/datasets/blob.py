import logging
from typing import Any, cast, Dict, Optional

import torch
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from lightkit.utils import PathType
from lightkit.data import DataLoader

from ._registry import register
from ._base import DataModule, OutputType
from ._utils import tabular_ood_dataset

logger = logging.getLogger(__name__)


@register("blob")
class BlobDataModule(DataModule):
    """
    Data module with two Gaussian blobs on the same Y-axis origin.
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
        logger.info("Preparing 'Blob Dataset'...")
        data_file_train = self.root / "blob" / "train.csv"
        data_file_test = self.root / "blob" / "test.csv"
        data_file_ood_val = self.root / "blob" / "ood_val.csv"
        data_file_ood_test = self.root / "blob" / "ood_test.csv"

        if not data_file_train.exists():
            logger.info("'Blob' could not be found locally. Creating new dataset...")
            data_file_train.parent.mkdir(parents=True, exist_ok=True)

            blobs = make_blobs(n_samples=1000, n_features=2, centers=[[-2, 0], [2, 0]], cluster_std=0.1)
            df = pd.DataFrame({"X":blobs[0][:, 0], "Y":blobs[0][:, 1], "class":blobs[1]})
            train, test = train_test_split(df, test_size=0.20, stratify=df.iloc[:, 2])
            train.to_csv(data_file_train, sep=",")
            test.to_csv(data_file_test, sep=",")

            ood_val = make_blobs(n_samples=200, centers=[[0, -2]], cluster_std=0.1)
            ood_val = pd.DataFrame({"X":ood_val[0][:, 0], "Y":ood_val[0][:, 1], "class":ood_val[1]})
            ood_val.to_csv(data_file_ood_val, sep=",")

            ood_test = make_blobs(n_samples=200, centers=[[0, 2]], cluster_std=0.1)
            ood_test = pd.DataFrame({"X":ood_test[0][:, 0], "Y":ood_test[0][:, 1], "class":ood_test[1]})
            ood_test.to_csv(data_file_ood_test, sep=",")

    def setup(self, stage: Optional[str] = None):
        if not self.did_setup:
            # Train, val, test dataset
            train_file = self.root / "blob" / "train.csv"
            test_file = self.root / "blob" / "test.csv"
            df = cast(pd.DataFrame, pd.read_csv(train_file, sep=",", index_col=0))
            df_test = cast(pd.DataFrame, pd.read_csv(test_file, sep=",", index_col=0))
            train, val = train_test_split(df, test_size=0.25, stratify=df.iloc[:, 2])

            X_train = torch.from_numpy(train.to_numpy()[:, :-1]).float()
            y_train = torch.from_numpy(train.to_numpy()[:, -1]).long()
            X_val = torch.from_numpy(val.to_numpy()[:, :-1]).float()
            y_val = torch.from_numpy(val.to_numpy()[:, -1]).long()
            X_test = torch.from_numpy(df_test.to_numpy()[:, :-1]).float()
            y_test = torch.from_numpy(df_test.to_numpy()[:, -1]).long()

            # OOD val dataset
            ood_val_file = self.root / "blob" / "ood_val.csv"
            ood_val = cast(pd.DataFrame, pd.read_csv(ood_val_file, sep=",", index_col=0))
            X_ood_val = torch.from_numpy(ood_val.to_numpy()[:, :-1]).float()

            # Initialize datasts
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)
            self.test_dataset = TensorDataset(X_test, y_test)
            # self.ood_val_dataset = TensorDataset(X_ood_val, y_ood_val)
            self.ood_val_dataset = tabular_ood_dataset(X_val, X_ood_val)
            self.did_setup = True
        
        if stage == "test" and not self.did_setup_ood:
            ood_test_file = self.root / "blob" / "ood_test.csv"
            ood_test = cast(pd.DataFrame, pd.read_csv(ood_test_file, sep=",", index_col=0))
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
        return [
            DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2),
        ]

    def ood_dataloaders(self):
        return {
            name: DataLoader(dataset, batch_size=self.batch_size, num_workers=2)
            for name, dataset in self.ood_test_datasets.items()
        }


@register("blob3")
class Blob3DataModule(DataModule):
    """
    Data module with three Gaussian blobs.
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
        return 3

    def prepare_data(self) -> None:
        logger.info("Preparing 'Blob Dataset'...")
        data_file_train = self.root / "blob3" / "train.csv"
        data_file_test = self.root / "blob3" / "test.csv"
        # data_file_ood_val = self.root / "blob3" / "ood_val.csv"
        data_file_ood_test = self.root / "blob3" / "ood_test.csv"

        if not data_file_train.exists():
            logger.info("'Blob' could not be found locally. Creating new dataset...")
            data_file_train.parent.mkdir(parents=True, exist_ok=True)

            blobs = make_blobs(n_samples=1000, n_features=3, centers=[[-2, -1], [2, -1], [0, 1]], cluster_std=0.1)
            df = pd.DataFrame({"X":blobs[0][:, 0], "Y":blobs[0][:, 1], "class":blobs[1]})
            train, test = train_test_split(df, test_size=0.20, stratify=df.iloc[:, 2])
            train.to_csv(data_file_train, sep=",")
            test.to_csv(data_file_test, sep=",")

            ood_test = make_blobs(n_samples=200, centers=[[0, -1]], cluster_std=0.1)
            ood_test = pd.DataFrame({"X":ood_test[0][:, 0], "Y":ood_test[0][:, 1], "class":ood_test[1]})
            ood_test.to_csv(data_file_ood_test, sep=",")

    def setup(self, stage: Optional[str] = None):
        if not self.did_setup:
            # Train, val, test dataset
            train_file = self.root / "blob3" / "train.csv"
            test_file = self.root / "blob3" / "test.csv"
            df = cast(pd.DataFrame, pd.read_csv(train_file, sep=",", index_col=0))
            df_test = cast(pd.DataFrame, pd.read_csv(test_file, sep=",", index_col=0))
            train, val = train_test_split(df, test_size=0.25, stratify=df.iloc[:, 2])

            X_train = torch.from_numpy(train.to_numpy()[:, :-1]).float()
            y_train = torch.from_numpy(train.to_numpy()[:, -1]).long()
            X_val = torch.from_numpy(val.to_numpy()[:, :-1]).float()
            y_val = torch.from_numpy(val.to_numpy()[:, -1]).long()
            X_test = torch.from_numpy(df_test.to_numpy()[:, :-1]).float()
            y_test = torch.from_numpy(df_test.to_numpy()[:, -1]).long()

            # Initialize datasts
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)
            self.test_dataset = TensorDataset(X_test, y_test)
            self.did_setup = True
        
        if stage == "test" and not self.did_setup_ood:
            ood_test_file = self.root / "blob3" / "ood_test.csv"
            ood_test = cast(pd.DataFrame, pd.read_csv(ood_test_file, sep=",", index_col=0))
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
        ]

    def test_dataloader(self):
        return [
            DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2),
        ]

    def ood_dataloaders(self):
        return {
            name: DataLoader(dataset, batch_size=self.batch_size, num_workers=2)
            for name, dataset in self.ood_test_datasets.items()
        }
