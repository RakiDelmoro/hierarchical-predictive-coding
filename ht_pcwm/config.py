from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    data_dir: str = "/workspaces/foundation-model-v2/dataset-mnist-moving"
    train_split: str = "train"
    val_split: str = "val"

    img_channels: int = 1
    img_height: int = 64
    img_width: int = 64
    sequence_length: int = 48

    batch_size: int = 32
    val_batch_size: int = 16
    num_workers: int = 4

    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    z1_channels: int = 128
    z2_channels: int = 128

    inference_steps: int = 5
    latent_update_alpha: float = 0.5

    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50

    lambda_frame: float = 1.0
    lambda_latent: float = 0.5
    lambda_reg: float = 0.1

    # Learned predictor settings
    predictor_hidden_channels: int = 64
    predictor_layers: int = 2
    predictor_clip_value: float = 1.0

    # Adaptive stopping settings
    adaptive_max_steps: int = 5
    adaptive_min_steps: int = 1
    adaptive_convergence_threshold: float = 1e-4

    device: str = "cuda"
    save_dir: str = "/workspaces/foundation-model-v2/ht_pcwm/checkpoints"
    log_dir: str = "/workspaces/foundation-model-v2/ht_pcwm/logs"

    viz_reconstruction_frames: int = 5
    viz_rollout_steps: int = 10
    viz_pca_samples: int = 2000
    viz_heatmap_frames: int = 20
    viz_heatmap_channels: int = 16


cfg = Config()


def update_cfg(**kwargs):
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
