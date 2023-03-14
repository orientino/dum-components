import logging
import sys
import zipfile
import os
from random import randint
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import torch
import numpy as np
import torchvision.datasets as tvd  # type: ignore
import torchvision.transforms as T  # type: ignore
from torch.utils.data import TensorDataset
from lightkit.data import DataLoader
from lightkit.utils import PathType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from wilds import get_dataset
from ._base import DataModule, OutputType
from ._registry import register
from ._utils import dataset_train_test_split, OodDataset, scale_oodom, TransformedDataset, TransformedDatasetWILDS

logger = logging.getLogger(__name__)


class _CifarDataModule(DataModule, ABC):
    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.did_setup = False
        self.did_setup_ood = False
        self.kwargs = dict(
            # num_workers=4,
            pin_memory=True,
        )

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 32, 32])

    @property
    @abstractmethod
    def _input_normalizer(self) -> T.Normalize:
        pass

    def prepare_data(self) -> None:
        logger.info("Preparing 'SVHN Train and Test'...")
        tvd.SVHN(str(self.root / "svhn"), split="test", download=True)
        tvd.SVHN(str(self.root / "svhn"), split="train", download=True)
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
        logger.info("Preparing 'Camelyon17 Train'...")
        self.camelyon = get_dataset("camelyon17", download=True, root_dir=str(self.root / "camelyon"))

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" and not self.did_setup_ood:
            train_ood_data = tvd.SVHN(
                str(self.root / "svhn"),
                split="train",
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )
            _, val_ood_dataset = dataset_train_test_split(
                train_ood_data, train_size=0.65, generator=self.generator
            )
            self.val_ood_dataset = val_ood_dataset
            self.val_oodom_dataset = TransformedDataset(
                val_ood_dataset,
                transform=T.Compose([T.Lambda(scale_oodom)]),
            )

            self.test_dataset_camelyon = TransformedDatasetWILDS(
                self.camelyon.get_subset(
                    "test",
                    transform=T.Compose([T.Resize([32, 32]), T.ToTensor(), self._input_normalizer]),
                )
            )

            self.ood_datasets = {
                "svhn_val": OodDataset(
                    self.val_dataset,
                    self.val_ood_dataset,
                ),
                "svhn": OodDataset(
                    self.test_dataset,
                    tvd.SVHN(
                        str(self.root / "svhn"),
                        split="test",
                        transform=T.Compose([T.ToTensor(), self._input_normalizer]),
                    ),
                ),
                "stl10": OodDataset(
                    self.test_dataset,
                    tvd.STL10(
                        str(self.root / "stl10"),
                        split="test",
                        transform=T.Compose([T.Resize([32, 32]), T.ToTensor(), self._input_normalizer]),
                    ),
                ),
                "celeba": OodDataset(
                    self.test_dataset,
                    tvd.CelebA(
                        str(self.root / "celeba"),
                        split="test",
                        transform=T.Compose([T.Resize([32, 32]), T.ToTensor(), self._input_normalizer]),
                    ),
                ),
                "camelyon": OodDataset(
                        self.test_dataset,
                        torch.utils.data.Subset(self.test_dataset_camelyon, range(10000)),
                ),
                "svhn_oodom": OodDataset(
                    self.test_dataset,
                    tvd.SVHN(
                        str(self.root / "svhn"),
                        split="test",
                        transform=T.Compose(
                            [T.ToTensor(), self._input_normalizer, T.Lambda(scale_oodom)]
                        ),
                    ),
                ),
            }

            # Mark done
            self.did_setup_ood = True

    def _setup_corrupted_datasets(self, root: PathType) -> None:
        trans = [
            'gaussian_noise', 
            'shot_noise', 
            'impulse_noise', 
            'defocus_blur',
            'glass_blur',
            'motion_blur', 
            'zoom_blur',
            'snow',   
            'frost',
            'fog',
            'brightness',
            'contrast',
            'elastic_transform',
            'pixelate',
            'jpeg_compression',
        ]

        # Load corrupted dataset
        self.corrupted_datasets = {}
        labels = torch.from_numpy(np.int64(np.load(os.path.join(root, 'labels.npy'))))
        for t in trans:
            images = torch.from_numpy(np.float32(np.load(os.path.join(root, t + '.npy')).transpose((0,3,1,2))))/255.

            # Divide per severity
            for i in range(5):
                self.corrupted_datasets[t+str(i+1)] = TransformedDataset(
                    TensorDataset(images[i*10000:(i+1)*10000], labels[i*10000:(i+1)*10000]),
                    transform=T.Compose([self._input_normalizer]),
                )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True, **self.kwargs)

    def val_dataloader(self) -> List[EVAL_DATALOADERS]:
        return [
            DataLoader(self.val_dataset, batch_size=256, **self.kwargs),
            DataLoader(self.val_ood_dataset, batch_size=256, **self.kwargs),
            # DataLoader(self.val_oodom_dataset, batch_size=256, **self.kwargs),
        ]

    def test_dataloader(self) -> List[EVAL_DATALOADERS]:
        return [
            DataLoader(self.test_dataset, batch_size=256),
        ]

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=256)
            for name, dataset in self.ood_datasets.items()
        }

    def ood_corrupted_dataloaders(self):
        return [
            DataLoader(dataset, batch_size=256) for dataset in list(self.corrupted_datasets.values())
        ]


@register("cifar10")
class Cifar10DataModule(_CifarDataModule):
    """
    Data module for the CIFAR-10 dataset.
    """

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def _input_normalizer(self) -> T.Normalize:
        return T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    def prepare_data(self) -> None:
        logger.info("Preparing 'CIFAR-10 Train and Test'...")
        tvd.CIFAR10(str(self.root / "cifar10"), train=True, download=True)
        tvd.CIFAR10(str(self.root / "cifar10"), train=False, download=True)
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            train_data = tvd.CIFAR10(
                str(self.root / "cifar10"),
                train=True,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )
            train_dataset, val_dataset = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )

            self.train_dataset = TransformedDataset(
                train_dataset,
                transform=T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(32, padding=4),
                    ]
                ),
            )
            self.val_dataset = val_dataset
            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.CIFAR10(
                str(self.root / "cifar10"),
                train=False,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )

        super().setup(stage=stage)
        # self._setup_corrupted_datasets(root=self.root / "cifar10-c/cifar-10-c")


@register("cifar100")
class Cifar100DataModule(_CifarDataModule):
    """
    Data module for the CIFAR-100 dataset.
    """

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def _input_normalizer(self) -> T.Normalize:
        return T.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])

    def prepare_data(self) -> None:
        logger.info("Preparing 'CIFAR-100 Train and Test'...")
        tvd.CIFAR100(str(self.root / "cifar100"), train=True, download=True)
        tvd.CIFAR100(str(self.root / "cifar100"), train=False, download=True)
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            train_data = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=True,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )
            train_dataset, val_dataset = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )

            self.train_dataset = TransformedDataset(
                train_dataset,
                transform=T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(32, padding=4),
                    ]
                ),
            )
            self.val_dataset = val_dataset
            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=False,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )

        super().setup(stage=stage)
        # self._setup_corrupted_datasets(root=self.root / "cifar100-c/cifar-100-c")


# ---------------------------------------------------------------------------------------------
# CIFAR100 module resized to size 224x224


class _Cifar100ResizedDataModule(Cifar100DataModule):
    """
    Data module for the CIFAR-100 dataset resized to 224x224.
    """

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=False,
                transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
            )

            train_ood_data = tvd.SVHN(
                str(self.root / "svhn"),
                split="train",
                transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
            )
            _, val_ood_dataset = dataset_train_test_split(
                train_ood_data, train_size=0.65, generator=self.generator
            )
            self.val_ood_dataset = val_ood_dataset
            self.val_oodom_dataset = TransformedDataset(
                val_ood_dataset,
                transform=T.Compose([T.Resize([224, 224]), T.Lambda(scale_oodom)]),
            )

            self.test_dataset_camelyon = TransformedDatasetWILDS(
                self.camelyon.get_subset(
                    "test",
                    transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
                )
            )

            self.ood_datasets = {
                "svhn_val": OodDataset(
                    self.val_dataset,
                    self.val_ood_dataset,
                ),
                "svhn": OodDataset(
                    self.test_dataset,
                    tvd.SVHN(
                        str(self.root / "svhn"),
                        split="test",
                        transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
                    ),
                ),
                "stl10": OodDataset(
                    self.test_dataset,
                    tvd.STL10(
                        str(self.root / "stl10"),
                        split="test",
                        transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
                    ),
                ),
                "celeba": OodDataset(
                    self.test_dataset,
                    tvd.CelebA(
                        str(self.root / "celeba"),
                        split="test",
                        transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
                    ),
                ),
                "camelyon": OodDataset(
                        self.test_dataset,
                        torch.utils.data.Subset(self.test_dataset_camelyon, range(10000)),
                ),
                "svhn_oodom": OodDataset(
                    self.test_dataset,
                    tvd.SVHN(
                        str(self.root / "svhn"),
                        split="test",
                        transform=T.Compose(
                            [T.Resize([224, 224]), T.ToTensor(), self._input_normalizer, T.Lambda(scale_oodom)]
                        ),
                    ),
                ),
            }

            # Mark done
            self.did_setup_ood = True


@register("cifar100-224")
class Cifar100ResizedDataModule(_Cifar100ResizedDataModule):
    """
    Data module for the CIFAR-100 dataset resized to 224x224.
    """

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            train_data = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=True,
                transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
            )
            train_dataset, val_dataset = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )

            self.train_dataset = TransformedDataset(
                train_dataset,
                transform=T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(224, padding=28),
                    ]
                ),
            )
            self.val_dataset = val_dataset
            self.did_setup = True
        
        super().setup(stage=stage)


@register("cifar100-224-partial")
class Cifar100ResizedPartialDataModule(_Cifar100ResizedDataModule):
    """
    Data module for the CIFAR-100 dataset resized to 224x224 and with only 10% of training set.
    """

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            train_data = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=True,
                transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
            )
            train_dataset, val_dataset = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )
            train_dataset, _ = dataset_train_test_split(
                train_dataset, train_size=0.1, generator=self.generator
            )

            self.train_dataset = TransformedDataset(
                train_dataset,
                transform=T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(224, padding=28),
                    ]
                ),
            )
            self.val_dataset = val_dataset
            self.did_setup = True

        super().setup(stage=stage)


# ---------------------------------------------------------------------------------------------
# CIFAR100 module with injected aleatoric noise
            

class _Cifar100NoiseDataModule(Cifar100DataModule):
    """
    Data module for the CIFAR-100 dataset with aleatoric noise.
    """

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            train_data = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=True,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )
            train_data = self._inject_aleatoric_noise(train_data, self.noise)
            train_dataset, val_dataset = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )

            self.train_dataset = TransformedDataset(
                train_dataset,
                transform=T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(32, padding=4),
                    ]
                ),
            )
            self.val_dataset = val_dataset
            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=False,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )

        super().setup(stage=stage)

    def _inject_aleatoric_noise(self, data, noise):
        """ 
        Inject artificial aleatoric noise to the dataset 
        """

        self.changed = []
        self.targets_old = data.targets
        self.targets_new = []

        for i, y in enumerate(self.targets_old):
            target = y
            if torch.rand(1)[0] < noise:
                self.changed.append(i)
                target = randint(0, self.num_classes-1)
            self.targets_new.append(target)
        data.targets = self.targets_new

        return data


@register("cifar100-noise1")
class Cifar100Noise1DataModule(_Cifar100NoiseDataModule):
    """
    Data module for the CIFAR-100 dataset with aleatoric noise.
    """

    @property
    def noise(self) -> float:
        return 0.1


@register("cifar100-noise2")
class Cifar100Noise2DataModule(_Cifar100NoiseDataModule):
    """
    Data module for the CIFAR-100 dataset with aleatoric noise.
    """

    @property
    def noise(self) -> float:
        return 0.2


# ---------------------------------------------------------------------------------------------
# CIFAR100 module with injected aleatoric noise


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.5):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


@register("cifar100-224-gaussian")
class Cifar100DataModule(_CifarDataModule):
    """
    Data module for the CIFAR-100 dataset with Gaussian noise
    """

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def _input_normalizer(self) -> T.Normalize:
        return T.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])

    def prepare_data(self) -> None:
        logger.info("Preparing 'CIFAR-100 Train and Test'...")
        tvd.CIFAR100(str(self.root / "cifar100"), train=True, download=True)
        tvd.CIFAR100(str(self.root / "cifar100"), train=False, download=True)
        # super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            train_data = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=True,
                transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
            )
            train_dataset, val_dataset = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )

            self.train_dataset = TransformedDataset(
                train_dataset,
                transform=T.Compose(
                    [
                        AddGaussianNoise(),
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(224, padding=28)
                    ]
                ),
            )
            self.val_dataset = val_dataset
            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=False,
                transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
            )

            train_ood_data = tvd.SVHN(
                str(self.root / "svhn"),
                split="train",
                transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
            )
            _, val_ood_dataset = dataset_train_test_split(
                train_ood_data, train_size=0.65, generator=self.generator
            )
            self.val_ood_dataset = val_ood_dataset
            self.val_oodom_dataset = TransformedDataset(
                val_ood_dataset,
                transform=T.Compose([T.Resize([224, 224]), T.Lambda(scale_oodom)]),
            )

            self.ood_datasets = {
                "svhn_val": OodDataset(
                    self.val_dataset,
                    self.val_ood_dataset,
                ),
                "svhn": OodDataset(
                    self.test_dataset,
                    tvd.SVHN(
                        str(self.root / "svhn"),
                        split="test",
                        transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
                    ),
                ),
                "stl10": OodDataset(
                    self.test_dataset,
                    tvd.STL10(
                        str(self.root / "stl10"),
                        split="test",
                        transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
                    ),
                ),
               "celeba": OodDataset(
                   self.test_dataset,
                   tvd.CelebA(
                       str(self.root / "celeba"),
                       split="test",
                       transform=T.Compose([T.Resize([224, 224]), T.ToTensor(), self._input_normalizer]),
                   ),
               ),
                "svhn_oodom": OodDataset(
                    self.test_dataset,
                    tvd.SVHN(
                        str(self.root / "svhn"),
                        split="test",
                        transform=T.Compose(
                            [T.Resize([224, 224]), T.ToTensor(), self._input_normalizer, T.Lambda(scale_oodom)]
                        ),
                    ),
                ),
            }

            # Mark done
            self.did_setup_ood = True
