from typing import Literal
from .tabular import TabularEncoder
from .wide_resnet import WideResNet
from .resnet import ResNet, ResNetDecoder
from .pretrained import tv_resnet18, tv_resnet50, tv_efficientnet_v2_s, tv_swin_t


EncoderType = Literal[
    "tabular", 
    "resnet18", 
    "wide-resnet",
    "resnet18-tv",
    "resnet50",
    "efficientnet_v2_s",
    "swin_t",
]


__all__ = [
    "EncoderType",
    "TabularEncoder",
    "WideResNet",
    "ResNet",
    "ResNetDecoder",
    "tv_resnet18",
    "tv_resnet50",
    "tv_efficientnet_v2_s",
    "tv_swin_t",
    "get_pretrained_path",
]
