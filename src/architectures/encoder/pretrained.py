import torch.nn as nn
from torchvision.models import resnet18, resnet50, efficientnet_v2_s, swin_t
from collections import OrderedDict


def tv_resnet18(pretrained, latent_dim, n_classes):
    """ ImageNet accuracy 69.758%, params 11.689.512 """

    weights = "IMAGENET1K_V1" if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model = nn.Sequential(OrderedDict([
        ("body", model),
        ("linear_latent", nn.Linear(512, latent_dim)),
        ("linear", nn.Linear(latent_dim, n_classes)),
        ("bn_out", nn.Identity()),
    ]))

    return model


def tv_resnet50(pretrained, latent_dim, n_classes):
    """ ImageNet accuracy 80.858%, params 25.557.032 """

    weights = "IMAGENET1K_V2" if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model = nn.Sequential(OrderedDict([
        ("body", model),
        ("linear_latent", nn.Linear(2048, latent_dim)),
        ("linear", nn.Linear(latent_dim, n_classes)),
        ("bn_out", nn.Identity()),
    ]))

    return model


def tv_efficientnet_v2_s(pretrained, latent_dim, n_classes):
    """ ImageNet accuracy 84.228%, params 21.458.488 """

    weights = "EfficientNet_V2_S_Weights.IMAGENET1K_V1" if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Identity()
    model = nn.Sequential(OrderedDict([
        ("body", model),
        ("linear_latent", nn.Linear(1280, latent_dim)),
        ("linear", nn.Linear(latent_dim, n_classes)),
        ("bn_out", nn.Identity()),
    ]))

    return model


def tv_swin_t(pretrained, latent_dim, n_classes):
    """ ImageNet accuracy 81.474%, params 28.288.354 """

    weights = "Swin_T_Weights.IMAGENET1K_V1" if pretrained else None
    model = swin_t(weights=weights)
    model.head = nn.Identity()
    model = nn.Sequential(OrderedDict([
        ("body", model),
        ("linear_latent", nn.Linear(768, latent_dim)),
        ("linear", nn.Linear(latent_dim, n_classes)),
        ("bn_out", nn.Identity()),
    ]))

    return model
