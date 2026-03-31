import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.transition import TransitionModel


class HierarchyLayer(nn.Module):
    def __init__(self, channels, spatial_size):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.transition = TransitionModel(channels)
        self.prediction = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, latent, latent_prediction=None):
        if latent_prediction is None:
            latent_prediction = latent
        latent_transition = self.transition(latent, latent_prediction)
        latent_hat = self.prediction(latent_transition)
        return latent_hat, latent_transition


class Hierarchy(nn.Module):
    def __init__(self, latent1_channels=128, latent2_channels=128, latent1_size=16, latent2_size=8):
        super().__init__()
        self.latent1_channels = latent1_channels
        self.latent2_channels = latent2_channels
        self.latent1_size = latent1_size
        self.latent2_size = latent2_size

        self.encoder = Encoder(in_channels=1, channel_list=[32, 64, 128], output_channels=latent1_channels)
        self.decoder = Decoder(in_channels=latent1_channels, channel_list=[64, 32], out_channels=1)

        self.latent1_predictor = nn.Sequential(
            nn.Conv2d(latent1_channels, latent1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent1_channels, latent1_channels, kernel_size=3, padding=1),
        )
        self.latent2_predictor = nn.Sequential(
            nn.Conv2d(latent2_channels, latent2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent2_channels, latent2_channels, kernel_size=3, padding=1),
        )

        self.downsample_latent1_to_latent2 = nn.Conv2d(latent1_channels, latent2_channels, kernel_size=4, stride=2, padding=1)
        self.upsample_latent2_to_latent1 = nn.ConvTranspose2d(latent2_channels, latent1_channels, kernel_size=4, stride=2, padding=1)

        self.latent2_transition = TransitionModel(latent2_channels)

    def encode(self, frame):
        return self.encoder(frame)

    def decode(self, latent1):
        return self.decoder(latent1)

    def predict_latent2(self, latent2_state, latent1_prediction=None):
        latent2_transition = self.latent2_transition(latent2_state, latent1_prediction)
        latent2_hat = self.latent2_predictor(latent2_transition)
        return latent2_hat, latent2_transition

    def predict_latent1_from_latent2(self, latent2):
        latent1_from_latent2 = self.upsample_latent2_to_latent1(latent2)
        latent1_hat = self.latent1_predictor(latent1_from_latent2)
        return latent1_hat

    def downsample_latent1(self, latent1):
        return self.downsample_latent1_to_latent2(latent1)
