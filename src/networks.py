import torch.nn as nn

from utils import ConvolutionalBlock, Downscaling, Upscaling


class UNet(nn.Module):
    """
    Implementation of the UNet network architecture. This is a modification of the
    network proposed https://arxiv.org/pdf/1505.04597.pdf in the context of
    semantic segmentation.

    This implementation is adapted to work as a surrogate model for simulating
    fluid flow against a cylinder depending on the cylinder's location and diameter.

    :param n_channels (int): number of channels in the input data.
    :param output_dim (int): dimension of the network's output.
    """

    def __init__(self, n_channels: int = 1, output_dim: int = 3):

        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.output_dim = output_dim

        self.input_conv = ConvolutionalBlock(n_channels, 8)
        self.downscaling_1 = Downscaling(8, 16)
        self.downscaling_2 = Downscaling(16, 32)
        self.downscaling_3 = Downscaling(32, 64)
        self.upscaling_1 = Upscaling(64, 32)
        self.upscaling_2 = Upscaling(32, 16)
        self.upscaling_3 = Upscaling(16, 8)
        self.output_conv = nn.Conv2d(8, self.output_dim, kernel_size=1)

    def forward(self, x):

        # Downscaling part of UNet
        x1 = self.input_conv(x)
        x2 = self.downscaling_1(x1)
        x3 = self.downscaling_2(x2)
        x = self.downscaling_3(x3)

        # Upscaling part of UNet
        x = self.upscaling_1(x, x3)
        x = self.upscaling_2(x, x2)
        x = self.upscaling_3(x, x1)
        output = self.output_conv(x)
        return output
