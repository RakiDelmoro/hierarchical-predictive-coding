import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=1, channel_list=[32, 64, 128], output_channels=128):
        super().__init__()
        layers = []
        for i, channels in enumerate(channel_list):
            in_ch = in_channels if i == 0 else channel_list[i - 1]
            layers.append(nn.Conv2d(in_ch, channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv2d(channel_list[-1], output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, frame):
        hidden = self.conv_layers(frame)
        latent = self.output_conv(hidden)
        return latent
