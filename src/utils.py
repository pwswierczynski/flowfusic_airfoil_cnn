import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalBlock(nn.Module):
    """
    Implementation of the main building block used in the UNet architecture.
    It consists of two repeated parts, each containing a convolutional part,
    batch normalization layer, and a ReLU nonlinearity.

    :params:
    in_channels (int): number of channels in the data
    out_channels (int): number of channels in the output
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3) -> None:

        super().__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x) -> torch.Tensor:
        return self.convolutional_block(x)


class Downscaling(nn.Module):
    """
    Downscaling building block of the UNet architecture.

    This block applies max-pooling operation and the convolutional block
    to the provided data.

    :params:
    in_channels (int): number of channels in the input data
    out_channels (int): number of channels in the output
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.downscale = nn.Sequential(
            nn.MaxPool2d(2), ConvolutionalBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downscale(x)


class Upscaling(nn.Module):
    """
    Upscaling building block of the UNet architecture.

    The main functionality of this block is applying
    convolutions to the first of the input layers, concatenating with the second layer,
    and applying another convolutional block to the output.

    :params:
    in_channels (int): number of channels in the input data
    out_channels (int): number of channels in the output
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.trans_conv = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.upsampling = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.conv_block = ConvolutionalBlock(int(3 * in_channels / 2), out_channels)

    def forward(self, first_layer, second_layer):

        current_output = self.upsampling(first_layer)

        diff_y = second_layer.size()[2] - current_output.size()[2]
        diff_x = second_layer.size()[3] - current_output.size()[3]

        current_output = F.pad(
            current_output,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        concatenated_layers = torch.cat([second_layer, current_output], dim=1)
        output = self.conv_block(concatenated_layers)

        return output
