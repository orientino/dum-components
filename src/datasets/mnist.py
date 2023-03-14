import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torchvision.datasets as tvd  # type: ignore
import torchvision.transforms as T  # type: ignore
from lightkit.data import DataLoader
from lightkit.utils import PathType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
from src.datasets._utils.ood import OodDataset
from ._base import DataModule, OutputType
from ._registry import register
from ._utils import dataset_train_test_split, scale_oodom, zero_channel

logger = logging.getLogger(__name__)


class _MnistDataModule(DataModule, ABC):
    train_dataset: Dataset[torch.Tensor]
    val_dataset: Dataset[torch.Tensor]
    test_dataset: Dataset[torch.Tensor]

    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.did_setup = False
        self.did_setup_ood = False

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([1, 28, 28])

    @property
    def num_classes(self) -> int:
        return 10

    @property
    @abstractmethod
    def _input_normalizer(self) -> T.Normalize:
        pass

    def prepare_data(self) -> None:
        logger.info("Preparing 'KMNIST'...")
        tvd.KMNIST(str(self.root / "kmnist"), train=False, download=True)
        logger.info("Preparing 'CIFAR-10'...")
        tvd.CIFAR10(str(self.root / "cifar10"), train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" and not self.did_setup_ood:
            self.ood_test_datasets["kmnist"] = OodDataset(
                self.test_dataset,
                tvd.KMNIST(
                    str(self.root / "kmnist"),
                    train=False,
                    transform=T.Compose([T.ToTensor(), self._input_normalizer]),
                ),
            )
            self.ood_test_datasets["cifar10_grayscale"] = OodDataset(
                self.test_dataset,
                tvd.CIFAR10(
                    str(self.root / "cifar10"),
                    train=False,
                    transform=T.Compose(
                        [T.Grayscale(), T.Resize([28, 28]), T.ToTensor(), self._input_normalizer]
                    ),
                ),
            )
            self.ood_test_datasets["kmnist_oodom"] = OodDataset(
                self.test_dataset,
                tvd.KMNIST(
                    str(self.root / "kmnist"),
                    train=False,
                    transform=T.Compose(
                        [T.ToTensor(), self._input_normalizer, T.Lambda(scale_oodom)]
                    ),
                ),
            )

            # Mark done
            self.did_setup_ood = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self) -> List[EVAL_DATALOADERS]:
        return [
            DataLoader(self.val_dataset, batch_size=512, num_workers=2),
        ]

    def test_dataloader(self) -> List[EVAL_DATALOADERS]:
        return [
            DataLoader(self.test_dataset, batch_size=512, num_workers=2),
        ]

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=512, num_workers=2)
            for name, dataset in self.ood_test_datasets.items()
        }


@register("mnist")
class MnistDataModule(_MnistDataModule):
    """
    Data module for the MNIST dataset.
    """

    @property
    def _input_normalizer(self) -> T.Normalize:
        return T.Normalize(mean=[0.1307], std=[0.3081])

    def prepare_data(self) -> None:
        logger.info("Preparing 'MNIST Train'...")
        tvd.MNIST(str(self.root / "mnist"), train=True, download=True)
        logger.info("Preparing 'MNIST Test'...")
        tvd.MNIST(str(self.root / "mnist"), train=False, download=True)
        logger.info("Preparing 'Fashion-MNIST'...")
        tvd.FashionMNIST(str(self.root / "fashion-mnist"), train=False, download=True)
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        transform = T.Compose([T.ToTensor(), self._input_normalizer])
        if not self.did_setup:
            train_data = tvd.MNIST(str(self.root / "mnist"), train=True, transform=transform)
            self.train_dataset, self.val_dataset = dataset_train_test_split(
                train_data,
                train_size=0.8,
                generator=self.generator,
            )
            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.MNIST(
                str(self.root / "mnist"), train=False, transform=transform
            )

        super().setup(stage)


@register("fashion-mnist")
class FashionMnistDataModule(_MnistDataModule):
    """
    Data module for the Fashion-MNIST dataset.
    """

    @property
    def _input_normalizer(self) -> T.Normalize:
        return T.Normalize(mean=[0.2860], std=[0.3530])

    def prepare_data(self) -> None:
        logger.info("Preparing 'Fashion-MNIST Train'...")
        tvd.FashionMNIST(str(self.root / "fashion-mnist"), train=True, download=True)
        logger.info("Preparing 'Fashion-MNIST Test'...")
        tvd.FashionMNIST(str(self.root / "fashion-mnist"), train=False, download=True)
        logger.info("Preparing 'MNIST'...")
        tvd.MNIST(str(self.root / "mnist"), train=False, download=True)
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        transform = T.Compose([T.ToTensor(), self._input_normalizer])
        if not self.did_setup:
            train_data = tvd.FashionMNIST(
                str(self.root / "fashion-mnist"), train=True, transform=transform
            )
            self.train_dataset, self.val_dataset = dataset_train_test_split(
                train_data,
                train_size=0.8,
                generator=self.generator,
            )
            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.FashionMNIST(
                str(self.root / "fashion-mnist"), train=False, transform=transform
            )

        super().setup(stage)


@register("cmnist")
class CMnistDataModule(_MnistDataModule):
    """
    Data module for the MNIST dataset with 3 channels to include Color MNIST.
    """

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 28, 28])

    @property
    def _input_normalizer(self) -> T.Normalize:
        return T.Normalize(mean=[0.1307], std=[0.3081])

    def prepare_data(self) -> None:
        logger.info("Preparing 'MNIST Train'...")
        tvd.MNIST(str(self.root / "mnist"), train=True, download=True)
        logger.info("Preparing 'MNIST Test'...")
        tvd.MNIST(str(self.root / "mnist"), train=False, download=True)
        logger.info("Preparing 'Fashion-MNIST'...")
        tvd.FashionMNIST(str(self.root / "fashion-mnist"), train=False, download=True)
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        transform = [
            T.ToTensor(), 
            self._input_normalizer,
            T.Lambda(lambda x: torch.cat([x, x, x], 0)),
        ]

        if not self.did_setup:
            train_data = tvd.MNIST(str(self.root / "mnist"), train=True, transform=T.Compose(transform))
            self.train_dataset, self.val_dataset = dataset_train_test_split(
                train_data,
                train_size=0.8,
                generator=self.generator,
            )
            self.val_ood_dataset = tvd.FashionMNIST(
                str(self.root / "fashion-mnist"),
                train=False,
                transform=T.Compose(transform),
            )
            self.did_setup = True

        # Setup test data
        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.MNIST(
                str(self.root / "mnist"), 
                train=False, 
                transform=T.Compose(transform),
            )
            self.test_dataset_cmnist = tvd.MNIST(
                str(self.root / "mnist"),
                train=False,
                transform=T.Compose(transform + [T.Lambda(zero_channel)]),
            )

        # Setup OOD data
        if stage == "test" and not self.did_setup_ood:
            self.ood_test_datasets = {
                "kmnist": OodDataset(
                    self.test_dataset,
                    tvd.KMNIST(
                        str(self.root / "kmnist"),
                        train=False,
                        transform=T.Compose(transform),
                    ),
                ),
                "cifar10": OodDataset(
                    self.test_dataset,
                    tvd.CIFAR10(
                        str(self.root / "cifar10"),
                        train=False,
                        transform=T.Compose([T.Resize([28, 28])] + transform[:-1]),
                    ),
                ),
                "cmnist": OodDataset(
                    self.test_dataset,
                    self.test_dataset_cmnist
                ),
                "kmnist_oodom": OodDataset(
                    self.test_dataset,
                    tvd.KMNIST(
                        str(self.root / "kmnist"),
                        train=False,
                        transform=T.Compose(transform + [T.Lambda(scale_oodom)]),
                    ),
                ),
            }

            # Mark done
            self.did_setup_ood = True

    def val_dataloader(self) -> List[EVAL_DATALOADERS]:
        return [
            DataLoader(self.val_dataset, batch_size=512),
            DataLoader(self.val_ood_dataset, batch_size=512),
        ]

    def test_dataloader(self) -> List[EVAL_DATALOADERS]:
        return [
            DataLoader(self.test_dataset, batch_size=512),
            DataLoader(self.test_dataset_cmnist, batch_size=512),
        ]


@register("cmnist-32")
class CMnist32DataModule(CMnistDataModule):
    """
    Data module for the MNIST dataset with 3 channels to include Color MNIST.
    Additionally, the size is scaled to 32x32.
    """

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 32, 32])

    def setup(self, stage: Optional[str] = None) -> None:
        transform = [
            T.Resize([32, 32]),
            T.ToTensor(), 
            self._input_normalizer,
            T.Lambda(lambda x: torch.cat([x, x, x], 0)),
        ]

        if not self.did_setup:
            train_data = tvd.MNIST(str(self.root / "mnist"), train=True, transform=T.Compose(transform))
            self.train_dataset, self.val_dataset = dataset_train_test_split(
                train_data,
                train_size=0.8,
                generator=self.generator,
            )
            self.val_ood_dataset = tvd.FashionMNIST(
                str(self.root / "fashion-mnist"),
                train=False,
                transform=T.Compose(transform),
            )
            self.did_setup = True

        # Setup test data
        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.MNIST(
                str(self.root / "mnist"), 
                train=False, 
                transform=T.Compose(transform),
            )
            self.test_dataset_cmnist = tvd.MNIST(
                str(self.root / "mnist"),
                train=False,
                transform=T.Compose(transform + [T.Lambda(zero_channel)]),
            )

        # Setup OOD data
        if stage == "test" and not self.did_setup_ood:
            self.ood_test_datasets = {
                "kmnist": OodDataset(
                    self.test_dataset,
                    tvd.KMNIST(
                        str(self.root / "kmnist"),
                        train=False,
                        transform=T.Compose(transform),
                    ),
                ),
                "cifar10": OodDataset(
                    self.test_dataset,
                    tvd.CIFAR10(
                        str(self.root / "cifar10"),
                        train=False,
                        transform=T.Compose(transform[:-1]),
                    ),
                ),
                "cmnist": OodDataset(
                    self.test_dataset,
                    self.test_dataset_cmnist
                ),
                "kmnist_oodom": OodDataset(
                    self.test_dataset,
                    tvd.KMNIST(
                        str(self.root / "kmnist"),
                        train=False,
                        transform=T.Compose(transform + [T.Lambda(scale_oodom)]),
                    ),
                ),
            }

            # Mark done
            self.did_setup_ood = True
