import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_channels=128, channel_list=[64, 32], out_channels=1):
        super().__init__()
        layers = []
        for i, channels in enumerate(channel_list):
            in_ch = in_channels if i == 0 else channel_list[i - 1]
            layers.append(nn.ConvTranspose2d(in_ch, channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(channel_list[-1], out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())
        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, latent):
        return self.deconv_layers(latent)
