from ._base import DataModule, OutputType
from ._registry import DATASET_REGISTRY
from .bike_sharing import BikeSharingNormalDataModule, BikeSharingPoissonDataModule
from .cifar import Cifar10DataModule, Cifar100DataModule
from .mnist import FashionMnistDataModule, MnistDataModule
from .nyu_depth_v2 import NyuDepthV2DataModule
from .sensorless_drive import SensorlessDriveDataModule
from .uci import ConcreteDataModule
from .moon import MoonDataModule
from .blob import BlobDataModule, Blob3DataModule
from .camelyon import CamelyonDataModule
from .weather import WeatherDataModule

__all__ = [
    "DataModule",
    "ConcreteDataModule",
    "DATASET_REGISTRY"
]
