import torch
import torch.nn as nn
from models.hierarchy import Hierarchy
from core.inference import run_inference_loop_with_adaptive_stopping
from core.energy import compute_energy
from core.learned_predictor import LearnedUpdatePredictor


class HTPCWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hierarchy = Hierarchy(
            latent1_channels=config.z1_channels,
            latent2_channels=config.z2_channels,
            latent1_size=16,
            latent2_size=8,
        )
        self.latent1_size = 16
        self.latent2_size = 8
        self.num_inference_steps = config.inference_steps
        self.inference_learning_rate = config.latent_update_alpha
        self.weight_frame = config.lambda_frame
        self.weight_latent = config.lambda_latent
        
        # Initialize learned predictor
        self.learned_predictor = LearnedUpdatePredictor(
            z1_channels=config.z1_channels,
            z2_channels=config.z2_channels,
            hidden_channels=config.predictor_hidden_channels,
            num_layers=config.predictor_layers
        )
        
        # Adaptive stopping settings
        self.adaptive_max_steps = config.adaptive_max_steps
        self.adaptive_min_steps = config.adaptive_min_steps
        self.adaptive_convergence_threshold = config.adaptive_convergence_threshold

    def forward(self, frame_current, frame_next):
        latent1 = self.hierarchy.encode(frame_current)
        latent2 = self.hierarchy.downsample_latent1(latent1)

        latent1_pred = latent1
        latent2_pred = latent2

        latent1_pred, latent2_pred, all_errors = run_inference_loop_with_adaptive_stopping(
            self.hierarchy, frame_current, frame_next, latent1, latent2, latent1_pred, latent2_pred,
            max_steps=self.adaptive_max_steps,
            min_steps=self.adaptive_min_steps,
            clip_value=self.config.predictor_clip_value,
            learned_predictor=self.learned_predictor,
            convergence_threshold=self.adaptive_convergence_threshold
        )

        frame_prediction = self.hierarchy.decode(latent1_pred)
        error_frame = frame_next - frame_prediction

        latent2_from_latent1 = self.hierarchy.upsample_latent2_to_latent1(latent1_pred)
        latent2_hat = self.hierarchy.latent2_predictor(self.hierarchy.latent2_transition(latent2))
        error_latent2 = latent2_pred - latent2_hat
        latent1_from_latent2 = self.hierarchy.upsample_latent2_to_latent1(latent2_pred)
        latent1_hat = self.hierarchy.latent1_predictor(latent1_from_latent2)
        error_latent1 = latent1_pred - latent1_hat

        energy_total, energy_frame, energy_latent1, energy_latent2 = compute_energy(
            error_frame, error_latent1, error_latent2,
            latent1_pred, latent2_pred,
            self.weight_frame, self.weight_latent, self.config.lambda_reg
        )

        return {
            "latent1": latent1,
            "latent2": latent2,
            "latent1_prediction": latent1_pred,
            "latent2_prediction": latent2_pred,
            "frame_prediction": frame_prediction,
            "error_frame": error_frame,
            "error_latent1": error_latent1,
            "error_latent2": error_latent2,
            "energy_total": energy_total,
            "energy_frame": energy_frame,
            "energy_latent1": energy_latent1,
            "energy_latent2": energy_latent2,
            "iterations_used": len(all_errors),
        }

    def rollout(self, frame_start, num_steps=10):
        predictions = []
        latent1 = self.hierarchy.encode(frame_start)
        latent2 = self.hierarchy.downsample_latent1(latent1)
        latent1_pred = latent1
        latent2_pred = latent2
        current_frame = frame_start

        for step in range(num_steps):
            latent1_pred, latent2_pred, _ = run_inference_loop_with_adaptive_stopping(
                self.hierarchy, current_frame, current_frame, latent1, latent2, latent1_pred, latent2_pred,
                max_steps=self.adaptive_max_steps,
                min_steps=self.adaptive_min_steps,
                clip_value=self.config.predictor_clip_value,
                learned_predictor=self.learned_predictor,
                convergence_threshold=self.adaptive_convergence_threshold
            )
            frame_prediction = self.hierarchy.decode(latent1_pred)
            predictions.append(frame_prediction)
            current_frame = frame_prediction.detach()
            latent1 = latent1_pred
            latent2 = latent2_pred

        return torch.stack(predictions, dim=1)
