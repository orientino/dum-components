# Follows:
# https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

import torch.nn as nn
import torch.nn.functional as F
from src.architectures.spectral import spectral_norm_conv, spectral_norm_fc, SpectralBatchNorm2d, SpectralBatchNorm1d


class WideBasic(nn.Module):
    def __init__(
        self,
        wrapped_conv,
        wrapped_batchnorm,
        input_size,
        in_c,
        out_c,
        stride,
        dropout_rate,
        residual=True,
    ):
        super().__init__()
        self.residual = residual

        self.bn1 = wrapped_batchnorm(in_c)
        self.conv1 = wrapped_conv(input_size, in_c, out_c, 3, stride)
        input_size = (input_size - 1) // stride + 1

        self.bn2 = wrapped_batchnorm(out_c)
        self.conv2 = wrapped_conv(input_size, out_c, out_c, 3, 1)

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        if stride != 1 or in_c != out_c:
            self.shortcut = wrapped_conv(input_size, in_c, out_c, 1, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        out += self.shortcut(x) if self.residual else 0

        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_classes,
        input_size,
        residual=True,
        spectral=(False, False, False),
        depth=28,
        widen_factor=10,
        dropout_rate=0.3,
        coeff=3,
        n_power_iterations=1,
        bn_out=False,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"

        # Define wrappers for spectral normalization
        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0
            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
            if spectral[1]:
                if kernel_size == 1:
                    # Use spectral norm fc, because bound are tight for 1x1 convolutions
                    return spectral_norm_fc(conv, coeff, n_power_iterations)
                else:
                    # Otherwise use spectral norm conv, with loose bound
                    input_dim = (in_c, input_size, input_size)
                    return spectral_norm_conv(conv, coeff, input_dim, n_power_iterations)
            return conv

        def wrapped_bn(num_features, dim=2):
            if dim == 2:
                if spectral[2]:
                    return SpectralBatchNorm2d(num_features, coeff) 
                else:
                    return nn.BatchNorm2d(num_features)
            elif dim == 1:
                if spectral[2]:
                    return SpectralBatchNorm1d(num_features, coeff)
                else:
                    return nn.BatchNorm1d(num_features)
            else:
                raise NotImplementedError

        self.wrapped_conv = wrapped_conv
        self.wrapped_bn = wrapped_bn
        self.residual = residual
        self.dropout_rate = dropout_rate

        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        strides = [1, 1, 2, 2]

        self.conv1 = wrapped_conv(input_size, 3, nStages[0], 3, strides[0])
        self.layer1, input_size = self._wide_layer(
            nStages[0:2], n, strides[1], input_size
        )
        self.layer2, input_size = self._wide_layer(
            nStages[1:3], n, strides[2], input_size
        )
        self.layer3, input_size = self._wide_layer(
            nStages[2:4], n, strides[3], input_size
        )

        self.bn1 = self.wrapped_bn(nStages[3])
        if spectral[0]:
            self.linear_latent = spectral_norm_fc(nn.Linear(nStages[3], latent_dim), coeff)
            self.linear = spectral_norm_fc(nn.Linear(latent_dim, num_classes), coeff)
        else:
            self.linear_latent = nn.Linear(nStages[3], latent_dim)
            self.linear = nn.Linear(latent_dim, num_classes)

        self.bn_out = None
        if bn_out:
            self.add_bn_output(latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def add_bn_output(self, output_dim):
        self.bn_out = self.wrapped_bn(output_dim, dim=1)

    def _wide_layer(self, channels, num_blocks, stride, input_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        in_c, out_c = channels

        for stride in strides:
            layers.append(
                WideBasic(
                    self.wrapped_conv,
                    self.wrapped_bn,
                    input_size,
                    in_c,
                    out_c,
                    stride,
                    self.dropout_rate,
                    self.residual,
                )
            )
            in_c = out_c
            input_size = (input_size - 1) // stride + 1

        return nn.Sequential(*layers), input_size

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.flatten(1)

        out = self.linear_latent(out)
        out = self.linear(out)

        # Add BN to the output
        if self.bn_out:
            out = self.bn_out(out)

        return out
