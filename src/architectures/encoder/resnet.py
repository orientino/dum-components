'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

This implementation (torch-like):
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

'''
import torch.nn as nn
import torch.nn.functional as F
from src.architectures.spectral import spectral_norm_conv, spectral_norm_conv_transposed, spectral_norm_fc, SpectralBatchNorm2d, SpectralBatchNorm1d


# ---------------------------------------------------------------------------------------------
# RESNET ENCODER


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, 
        wrapped_conv,
        wrapped_bn,
        in_c, 
        out_c, 
        stride,
        input_size,
        residual=True,
    ):
        super(BasicBlock, self).__init__()
        self.residual = residual

        self.conv1 = wrapped_conv(input_size, in_c, out_c, 3, stride)
        self.bn1 = wrapped_bn(out_c)
        input_size = (input_size - 1) // stride + 1

        self.conv2 = wrapped_conv(input_size, out_c, out_c, 3, 1)
        self.bn2 = wrapped_bn(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != self.expansion*out_c:
            self.shortcut = nn.Sequential(
                wrapped_conv(input_size, in_c, self.expansion*out_c, 1, stride),
                wrapped_bn(self.expansion*out_c),
            )
        else: 
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) if self.residual else 0
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, 
        latent_dim,
        output_dim, 
        input_size,
        block=BasicBlock, 
        num_blocks=[2, 2, 2, 2], 
        residual=True,
        spectral=(False, False, False),     # (spectral_fc, spectral_conv, spectral_bn)
        lipschitz_constant=3, 
        n_power_iterations=1,
        bn_out=False,
        reconst_out=False,
    ):
        super(ResNet, self).__init__()

        # Define wrappers for spectral normalization
        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0
            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
            if spectral[1]:
                if kernel_size == 1:
                    # Use spectral norm fc, because bound are tight for 1x1 convolutions
                    return spectral_norm_fc(conv, lipschitz_constant, n_power_iterations)
                else:
                    # Otherwise use spectral norm conv, with loose bound
                    input_dim = (in_c, input_size, input_size)
                    return spectral_norm_conv(conv, lipschitz_constant, input_dim, n_power_iterations)
            return conv

        def wrapped_bn(num_features, dim=2):
            if dim == 2:
                if spectral[2]:
                    return SpectralBatchNorm2d(num_features, lipschitz_constant) 
                else:
                    return nn.BatchNorm2d(num_features)
            elif dim == 1:
                if spectral[2]:
                    return SpectralBatchNorm1d(num_features, lipschitz_constant)
                else:
                    return nn.BatchNorm1d(num_features)
            else:
                raise NotImplementedError
        
        self.wrapped_conv = wrapped_conv
        self.wrapped_bn = wrapped_bn
        self.residual = residual

        # Define the architecture
        strides = [1, 2, 2, 2]
        self.conv1 = wrapped_conv(input_size, in_c=3, out_c=32, kernel_size=3, stride=1)
        self.bn1 = wrapped_bn(32)
        self.layer1, input_size = self._make_layer(
            block, 32, 32, num_blocks[0], strides[0], input_size
        )
        self.layer2, input_size = self._make_layer(
            block, 32, 64, num_blocks[1], strides[1], input_size
        )
        self.layer3, input_size = self._make_layer(
            block, 64, 128, num_blocks[2], strides[2], input_size
        )
        self.layer4, input_size = self._make_layer(
            block, 128, 256, num_blocks[3], strides[3], input_size
        )
        if spectral[0]:
            self.linear_latent = spectral_norm_fc(nn.Linear(256 * block.expansion, latent_dim), lipschitz_constant) 
            self.linear = spectral_norm_fc(nn.Linear(latent_dim, output_dim), lipschitz_constant) 
        else:
            self.linear_latent = nn.Linear(256 * block.expansion, latent_dim)
            self.linear = nn.Linear(latent_dim, output_dim)

        self.reconst_out = reconst_out
        self.bn_out = None
        if bn_out:
            self.add_bn_output(output_dim)

    def _make_layer(self, block, in_c, out_c, num_blocks, stride, input_size):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.wrapped_conv, 
                    self.wrapped_bn, 
                    in_c, 
                    out_c, 
                    stride,
                    input_size,
                    self.residual,
                )
            )
            in_c = out_c * block.expansion
            input_size = (input_size - 1) // stride + 1
        return nn.Sequential(*layers), input_size

    def add_bn_output(self, output_dim):
        self.bn_out = self.wrapped_bn(output_dim, dim=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        rec = out

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear_latent(out) 
        out = self.linear(out)

        # Add BN to the output
        if self.bn_out:
            out = self.bn_out(out)

        # Return the input for the reconstruction decoder
        if self.reconst_out and self.training:
            return out, rec

        return out


# ---------------------------------------------------------------------------------------------
# RESNET DECODER


class DecoderBlock(nn.Module):
    def __init__(
        self, 
        wrapped_conv_transposed,
        wrapped_bn,
        in_c, 
        out_c, 
        stride,
        input_size,
    ):
        super(DecoderBlock, self).__init__()

        self.conv_transposed1 = wrapped_conv_transposed(input_size, in_c, in_c, 3, stride)
        self.bn1 = wrapped_bn(in_c)
        self.conv_transposed2 = wrapped_conv_transposed(input_size, in_c, out_c, 3, stride)
        self.bn2 = wrapped_bn(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                wrapped_conv_transposed(input_size, in_c, out_c, 1, stride),
                wrapped_bn(out_c),
            )
        else: 
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_transposed1(x)))
        out = self.bn2(self.conv_transposed2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetDecoder(nn.Module):
    def __init__(
        self, 
        spectral=(False, False, False),
        lipschitz_constant=3,
        n_power_iterations=1,
    ):
        super(ResNetDecoder, self).__init__()

        # Define wrappers for spectral normalization
        def wrapped_conv_transposed(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0
            conv = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding, bias=False)
            if spectral[1]:
                if kernel_size == 1:
                    # Use spectral norm fc, because bound are tight for 1x1 convolutions
                    return spectral_norm_fc(conv, lipschitz_constant, n_power_iterations)
                else:
                    # Otherwise use spectral norm conv, with loose bound
                    input_dim = (in_c, input_size, input_size)
                    return spectral_norm_conv_transposed(conv, lipschitz_constant, input_dim, n_power_iterations)
            return conv

        def wrapped_bn(num_features):
            if spectral[2]:
                return SpectralBatchNorm2d(num_features, lipschitz_constant) 
            return nn.BatchNorm2d(num_features)

        # Define the architecture
        self.upsample = nn.Upsample(scale_factor=2)
        self.layer1 = DecoderBlock(wrapped_conv_transposed, wrapped_bn, 256, 128, 1, input_size=4)
        self.layer2 = DecoderBlock(wrapped_conv_transposed, wrapped_bn, 128, 64, 1, input_size=8)
        self.layer3 = DecoderBlock(wrapped_conv_transposed, wrapped_bn, 64, 32, 1, input_size=16)
        self.layer4 = DecoderBlock(wrapped_conv_transposed, wrapped_bn, 32, 3, 1, input_size=32)

    def forward(self, x):
        out = self.layer1(x)
        out = self.upsample(out)
        # print("upsample: " + str(out.size()))

        out = self.layer2(out)
        out = self.upsample(out)
        # print("upsample: " + str(out.size()))

        out = self.layer3(out)
        out = self.upsample(out)
        # print("upsample: " + str(out.size()))

        out = self.layer4(out)
        # print("upsample: " + str(out.size()))
        return out
