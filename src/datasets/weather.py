import logging
from typing import Any, cast, Dict, Optional

import torch
import pandas as pd
from torch.utils.data import TensorDataset
from lightkit.utils import PathType
from lightkit.data import DataLoader

from ._registry import register
from ._base import DataModule, OutputType
# from ._utils import scale_oodom, StandardScaler, tabular_ood_dataset, tabular_train_test_split
from ._utils import scale_oodom, tabular_ood_dataset, tabular_train_test_split

logger = logging.getLogger(__name__)


@register("weather")
class WeatherDataModule(DataModule):
    """
    """

    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        super().__init__(root, seed)
        self.did_setup = False
        self.did_setup_ood = False
        self.kwargs = dict(
            # num_workers=4,
            batch_size=1024,
        )

    @property
    def output_type(self) -> OutputType:
        return "normal"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([122])

    def prepare_data(self) -> None:
        logger.info("Preparing 'Weather Dataset'...")

    def setup(self, stage: Optional[str] = None):
        if not self.did_setup:
            # Read the splitted data
            file_train = self.root / "weather" / "shifts_canonical_train.csv"
            file_val_id = self.root / "weather" / "shifts_canonical_dev_in.csv"
            file_val_ood = self.root / "weather" / "shifts_canonical_dev_out.csv"
            file_test_id = self.root / "weather" / "shifts_canonical_eval_in.csv"
            file_test_ood = self.root / "weather" / "shifts_canonical_eval_out.csv"
            df_train = cast(pd.DataFrame, pd.read_csv(file_train, sep=",", index_col=0, nrows=10000))
            df_val_id = cast(pd.DataFrame, pd.read_csv(file_val_id, sep=",", index_col=0, nrows=10000))
            df_val_ood = cast(pd.DataFrame, pd.read_csv(file_val_ood, sep=",", index_col=0, nrows=10000))
            df_test_id = cast(pd.DataFrame, pd.read_csv(file_test_id, sep=",", index_col=0, nrows=10000))
            df_test_ood = cast(pd.DataFrame, pd.read_csv(file_test_ood, sep=",", index_col=0, nrows=10000))

            # Drop NA data
            df_train.dropna(inplace=True)
            df_val_id.dropna(inplace=True)
            df_val_ood.dropna(inplace=True)
            df_test_id.dropna(inplace=True)
            df_test_ood.dropna(inplace=True)

            # Normalize data
            # X_train = df_train.iloc[:, 6:].to_numpy()
            # y_train = df_train["fact_temperature"].to_numpy()

            # from sklearn.preprocessing import StandardScaler
            # input_scaler, output_scaler = StandardScaler(), StandardScaler()
            # input_scaler.fit(X_train)
            # output_scaler.fit(y_train.reshape(-1, 1))

            # X_train = torch.from_numpy(input_scaler.transform(X_train)).float()
            # y_train = torch.from_numpy(output_scaler.transform(y_train.reshape(-1, 1))).float()
            # X_val_id = input_scaler.transform(torch.from_numpy(df_val_id.iloc[:, 6:].to_numpy()).float())
            # y_val_id = output_scaler.transform(torch.from_numpy(df_val_id["fact_temperature"].to_numpy()).float())
            # X_val_ood = input_scaler.transform(torch.from_numpy(df_val_ood.iloc[:, 6:].to_numpy()).float())
            # y_val_ood = output_scaler.transform(torch.from_numpy(df_val_ood["fact_temperature"].to_numpy()).float())
            # X_test_id = input_scaler.transform(torch.from_numpy(df_test_id.iloc[:, 6:].to_numpy()).float())
            # y_test_id = output_scaler.transform(torch.from_numpy(df_test_id["fact_temperature"].to_numpy()).float())
            # X_test_ood = input_scaler.transform(torch.from_numpy(df_test_ood.iloc[:, 6:].to_numpy()).float())
            # y_test_ood = output_scaler.transform(torch.from_numpy(df_test_ood["fact_temperature"].to_numpy()).float())

            X_train = torch.from_numpy(df_train.iloc[:, 6:].to_numpy()).float()
            y_train = torch.from_numpy(df_train["fact_temperature"].to_numpy()).float()
            X_val_id = torch.from_numpy(df_val_id.iloc[:, 6:].to_numpy()).float()
            y_val_id = torch.from_numpy(df_val_id["fact_temperature"].to_numpy()).float()

            # Initialize datasts
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_id_dataset = TensorDataset(X_val_id, y_val_id)
            # self.val_ood_dataset = TensorDataset(X_val_ood, y_val_ood)
            # self.test_id_dataset = TensorDataset(X_test_id, y_test_id)
            # self.test_ood_dataset = TensorDataset(X_test_ood, y_test_ood)
        
            # self.ood_test_datasets = {
            #     "weather": tabular_ood_dataset(X_test_id, X_test_ood)
            # }

            self.did_setup = True
            self.did_setup_ood = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, pin_memory=True, **self.kwargs)

    def val_dataloader(self):
        return [
            DataLoader(self.val_id_dataset, pin_memory=True, **self.kwargs),
            # DataLoader(self.val_ood_dataset, pin_memory=True, **self.kwargs),
        ]

    def test_dataloader(self):
        return [
            DataLoader(self.test_id_dataset, **self.kwargs),
            DataLoader(self.test_ood_dataset, **self.kwargs),
        ]

    def ood_dataloaders(self):
        return {
            name: DataLoader(dataset, **self.kwargs)
            for name, dataset in self.ood_test_datasets.items()
        }
