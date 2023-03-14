from ._base import DataModule, OutputType
from ._registry import DATASET_REGISTRY
from .bike_sharing import BikeSharingNormalDataModule, BikeSharingPoissonDataModule
from .cifar import Cifar10DataModule, Cifar100DataModule
from .mnist import FashionMnistDataModule, MnistDataModule
from .moon import MoonDataModule
from .blob import BlobDataModule, Blob3DataModule
from .camelyon import CamelyonDataModule

__all__ = [
    "DataModule",
    "ConcreteDataModule",
    "DATASET_REGISTRY"
]
