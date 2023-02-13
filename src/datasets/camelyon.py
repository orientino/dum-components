import logging
from typing import Any, Dict, Optional, List
import torch
import torchvision.transforms as T  # type: ignore
import torchvision.datasets as tvd  # type: ignore

from wilds import get_dataset
from lightkit.data import DataLoader
from lightkit.utils import PathType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from ._base import DataModule, OutputType
from ._registry import register
from ._utils import TransformedDatasetWILDS, dataset_train_test_split, OodDataset, scale_oodom

logger = logging.getLogger(__name__)


@register("camelyon")
class CamelyonDataModule(DataModule):
    def __init__(self, root: Optional[PathType] = None, batch_size: int = 32, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.batch_size = batch_size
        self.kwargs = dict(
            # num_workers=4,
            pin_memory=True,
        )

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 96, 96])

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def _input_normalizer(self) -> T.Normalize:
        # return T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        pass

    def prepare_data(self) -> None:
        logger.info("Preparing 'Camelyon17 Train'...")
        self.dataset = get_dataset("camelyon17", download=True, root_dir=str(self.root / "camelyon"))
        logger.info("Preparing 'SVHN Test'...")
        tvd.SVHN(str(self.root / "svhn"), split="test", download=True)
        logger.info("Preparing 'STL10 Test'...")
        tvd.STL10(str(self.root / "stl10"), split="test", download=True)
        try:
            logger.info("Preparing 'CelebA'...")
            tvd.CelebA(str(self.root / "celeba"), split="test", download=True)
        except zipfile.BadZipFile:
            logger.error(
                "Downloading 'CelebA' failed due to download restrictions on Google Drive. "
                "Please download manually from https://drive.google.com/drive/folders/"
                "0B7EVK8r0v71pWEZsZE9oNnFzTm8 and put the files into '%s'.",
                self.root / "celeba",
            )
            sys.exit(1)

    def setup(self, stage: Optional[str] = None) -> None:
        train_data = self.dataset.get_subset(
            "train",
            transform=T.Compose(
                [T.ToTensor()]
            ),
        )

        self.train_dataset = TransformedDatasetWILDS(
            train_data,
            transform=T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                ]
            )
        )
        self.val_id_dataset = TransformedDatasetWILDS(
            self.dataset.get_subset(
                "id_val",
                transform=T.Compose(
                    [T.ToTensor()]
                ),
            )
        )
        self.val_ood_dataset = TransformedDatasetWILDS(
            self.dataset.get_subset(
                "val",
                transform=T.Compose(
                    [T.ToTensor()]
                ),
            )
        )
        self.test_dataset = TransformedDatasetWILDS(
            self.dataset.get_subset(
                "test",
                transform=T.Compose(
                    [T.ToTensor()]
                ),
            )
        )

        train_svhn_ood_data = tvd.SVHN(
            str(self.root / "svhn"),
            split="train",
            transform=T.Compose([T.Resize([96, 96]), T.ToTensor()]),
        )
        _, val_svhn_ood_dataset = dataset_train_test_split(
            train_svhn_ood_data, train_size=0.65, generator=self.generator
        )
        self.val_svhn_ood_dataset = val_svhn_ood_dataset

        n_test_ood = 10000
        self.ood_datasets = {
            "svhn_val": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                self.val_svhn_ood_dataset,
            ),
            "svhn_val_ood": OodDataset(
                torch.utils.data.Subset(self.val_ood_dataset, range(n_test_ood)),
                self.val_svhn_ood_dataset,
            ),
            "svhn": OodDataset(
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
                tvd.SVHN(
                    str(self.root / "svhn"),
                    split="test",
                    transform=T.Compose([T.Resize([96, 96]), T.ToTensor()]),
                ),
            ),
            "stl10": OodDataset(
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
                tvd.STL10(
                    str(self.root / "stl10"),
                    split="test",
                    transform=T.Compose([T.Resize([96, 96]), T.ToTensor()]),
                ),
            ),
            "celeba": OodDataset(
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
                tvd.CelebA(
                    str(self.root / "celeba"),
                    split="test",
                    transform=T.Compose([T.Resize([96, 96]), T.ToTensor()]),
                ),
            ),
            "camelyon": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
            ),
            "svhn_oodom": OodDataset(
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
                tvd.SVHN(
                    str(self.root / "svhn"),
                    split="test",
                    transform=T.Compose(
                        [T.Resize([96, 96]), T.ToTensor(), T.Lambda(scale_oodom)]
                    ),
                ),
            ),
            "svhn_id": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                tvd.SVHN(
                    str(self.root / "svhn"),
                    split="test",
                    transform=T.Compose([T.Resize([96, 96]), T.ToTensor()]),
                ),
            ),
            "stl10_id": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                tvd.STL10(
                    str(self.root / "stl10"),
                    split="test",
                    transform=T.Compose([T.Resize([96, 96]), T.ToTensor()]),
                ),
            ),
            "celeba_id": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                tvd.CelebA(
                    str(self.root / "celeba"),
                    split="test",
                    transform=T.Compose([T.Resize([96, 96]), T.ToTensor()]),
                ),
            ),
            "svhn_oodom_id": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                tvd.SVHN(
                    str(self.root / "svhn"),
                    split="test",
                    transform=T.Compose(
                        [T.Resize([96, 96]), T.ToTensor(), T.Lambda(scale_oodom)]
                    ),
                ),
            ),
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            **self.kwargs
        )

    def val_dataloader(self) -> List[EVAL_DATALOADERS]:
        return [
            DataLoader(self.val_id_dataset, batch_size=self.batch_size, pin_memory=True),
            DataLoader(self.val_ood_dataset, batch_size=self.batch_size, pin_memory=True),
        ]

    def test_dataloader(self) -> List[EVAL_DATALOADERS]:
        # Re-evaluate validation results, since we re-load the model based on the best checkpoint
        # val_id_dataset is saved as test/id/...
        # val_ood_dataset is saved as test/ood/...
        # test_dataset is saved as test/oodom/...
        return [
            DataLoader(self.val_id_dataset, batch_size=self.batch_size),
            DataLoader(self.val_ood_dataset, batch_size=self.batch_size),
            DataLoader(self.test_dataset, batch_size=self.batch_size),
        ]

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=self.batch_size)
            for name, dataset in self.ood_datasets.items()
        }


@register("camelyon-224")
class CamelyonDataModule(DataModule):
    def __init__(self, root: Optional[PathType] = None, batch_size: int = 32, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.batch_size = batch_size
        self.kwargs = dict(
            # num_workers=4,
            pin_memory=True,
        )

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 224, 224])

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def _input_normalizer(self) -> T.Normalize:
        # return T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        pass

    def prepare_data(self) -> None:
        logger.info("Preparing 'Camelyon17 Train'...")
        self.dataset = get_dataset("camelyon17", download=True, root_dir=str(self.root / "camelyon"))
        logger.info("Preparing 'SVHN Test'...")
        tvd.SVHN(str(self.root / "svhn"), split="test", download=True)
        logger.info("Preparing 'STL10 Test'...")
        tvd.STL10(str(self.root / "stl10"), split="test", download=True)
        try:
            logger.info("Preparing 'CelebA'...")
            tvd.CelebA(str(self.root / "celeba"), split="test", download=True)
        except zipfile.BadZipFile:
            logger.error(
                "Downloading 'CelebA' failed due to download restrictions on Google Drive. "
                "Please download manually from https://drive.google.com/drive/folders/"
                "0B7EVK8r0v71pWEZsZE9oNnFzTm8 and put the files into '%s'.",
                self.root / "celeba",
            )
            sys.exit(1)

    def setup(self, stage: Optional[str] = None) -> None:
        train_data = self.dataset.get_subset(
            "train",
            transform=T.Compose(
                [T.Resize([224, 224]), T.ToTensor()]
            ),
        )

        self.train_dataset = TransformedDatasetWILDS(
            train_data,
            transform=T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                ]
            )
        )
        self.val_id_dataset = TransformedDatasetWILDS(
            self.dataset.get_subset(
                "id_val",
                transform=T.Compose(
                    [T.Resize([224, 224]), T.ToTensor()]
                ),
            )
        )
        self.val_ood_dataset = TransformedDatasetWILDS(
            self.dataset.get_subset(
                "val",
                transform=T.Compose(
                    [T.Resize([224, 224]), T.ToTensor()]
                ),
            )
        )
        self.test_dataset = TransformedDatasetWILDS(
            self.dataset.get_subset(
                "test",
                transform=T.Compose(
                    [T.Resize([224, 224]), T.ToTensor()]
                ),
            )
        )

        train_svhn_ood_data = tvd.SVHN(
            str(self.root / "svhn"),
            split="train",
            transform=T.Compose([T.Resize([224, 224]), T.ToTensor()]),
        )
        _, val_svhn_ood_dataset = dataset_train_test_split(
            train_svhn_ood_data, train_size=0.65, generator=self.generator
        )
        self.val_svhn_ood_dataset = val_svhn_ood_dataset

        n_test_ood = 10000
        self.ood_datasets = {
            "svhn_val": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                self.val_svhn_ood_dataset,
            ),
            "svhn_val_ood": OodDataset(
                torch.utils.data.Subset(self.val_ood_dataset, range(n_test_ood)),
                self.val_svhn_ood_dataset,
            ),
            "svhn": OodDataset(
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
                tvd.SVHN(
                    str(self.root / "svhn"),
                    split="test",
                    transform=T.Compose([T.Resize([224, 224]), T.ToTensor()]),
                ),
            ),
            "stl10": OodDataset(
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
                tvd.STL10(
                    str(self.root / "stl10"),
                    split="test",
                    transform=T.Compose([T.Resize([224, 224]), T.ToTensor()]),
                ),
            ),
            "celeba": OodDataset(
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
                tvd.CelebA(
                    str(self.root / "celeba"),
                    split="test",
                    transform=T.Compose([T.Resize([224, 224]), T.ToTensor()]),
                ),
            ),
            "camelyon": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
            ),
            "svhn_oodom": OodDataset(
                torch.utils.data.Subset(self.test_dataset, range(n_test_ood)),
                tvd.SVHN(
                    str(self.root / "svhn"),
                    split="test",
                    transform=T.Compose(
                        [T.Resize([224, 224]), T.ToTensor(), T.Lambda(scale_oodom)]
                    ),
                ),
            ),
            "svhn_id": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                tvd.SVHN(
                    str(self.root / "svhn"),
                    split="test",
                    transform=T.Compose([T.Resize([224, 224]), T.ToTensor()]),
                ),
            ),
            "stl10_id": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                tvd.STL10(
                    str(self.root / "stl10"),
                    split="test",
                    transform=T.Compose([T.Resize([224, 224]), T.ToTensor()]),
                ),
            ),
            "celeba_id": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                tvd.CelebA(
                    str(self.root / "celeba"),
                    split="test",
                    transform=T.Compose([T.Resize([224, 224]), T.ToTensor()]),
                ),
            ),
            "svhn_oodom_id": OodDataset(
                torch.utils.data.Subset(self.val_id_dataset, range(n_test_ood)),
                tvd.SVHN(
                    str(self.root / "svhn"),
                    split="test",
                    transform=T.Compose(
                        [T.Resize([224, 224]), T.ToTensor(), T.Lambda(scale_oodom)]
                    ),
                ),
            ),
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            **self.kwargs
        )

    def val_dataloader(self) -> List[EVAL_DATALOADERS]:
        return [
            DataLoader(self.val_id_dataset, batch_size=self.batch_size, pin_memory=True),
            DataLoader(self.val_ood_dataset, batch_size=self.batch_size, pin_memory=True),
        ]

    def test_dataloader(self) -> List[EVAL_DATALOADERS]:
        # Re-evaluate validation results, since we re-load the model based on the best checkpoint
        # val_id_dataset is saved as test/id/...
        # val_ood_dataset is saved as test/ood/...
        # test_dataset is saved as test/oodom/...
        return [
            DataLoader(self.val_id_dataset, batch_size=self.batch_size),
            DataLoader(self.val_ood_dataset, batch_size=self.batch_size),
            DataLoader(self.test_dataset, batch_size=self.batch_size),
        ]

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=self.batch_size)
            for name, dataset in self.ood_datasets.items()
        }
