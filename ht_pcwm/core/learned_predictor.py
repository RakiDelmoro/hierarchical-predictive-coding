import torch
import torch.nn as nn


class LearnedUpdatePredictor(nn.Module):
    """
    Full learned update predictor for predictive coding inference.
    Takes context (current latents, errors, frames) and predicts updates.
    """
    
    def __init__(self, z1_channels=128, z2_channels=128, hidden_channels=64, num_layers=2):
        super().__init__()
        self.z1_channels = z1_channels
        self.z2_channels = z2_channels
        
        # Input: z1_pred, z2_pred, error_z1, error_z2, frame_current, frame_next
        # Frame has 1 channel (grayscale), z1/z2 have z1_channels/z2_channels each
        input_channels = z1_channels * 2 + z2_channels * 2 + 2  # z1+z2 latents + errors + 2 frames
        
        # Build predictor network (1-2 layer CNN as requested for stability)
        layers = []
        in_ch = input_channels
        
        for i in range(num_layers - 1):
            out_ch = hidden_channels * (2 ** i)  # 64, 128, 256...
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ])
            in_ch = out_ch
        
        # Final layer outputs updates for both z1 and z2
        # z1 update: same spatial size as z1 (16×16)
        # z2 update: same spatial size as z2 (8×8)
        # We'll predict both in a single pass by having two heads
        
        # Shared body
        self.shared_body = nn.Sequential(*layers)
        
        # z1 update head (outputs z1_channels updates at z1 spatial size)
        self.z1_update_head = nn.Sequential(
            nn.Conv2d(in_ch, z1_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Bound outputs to [-1, 1]
        )
        
        # z2 update head (outputs z2_channels updates at z1 spatial size, then downsample)
        self.z2_update_head = nn.Sequential(
            nn.Conv2d(in_ch, z2_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Bound outputs to [-1, 1]
        )
        
        # Downsample layer for z2 (from z1 spatial size to z2 spatial size)
        self.downsample_z2 = nn.Conv2d(z2_channels, z2_channels, kernel_size=3, stride=2, padding=1)
        
        # Initialize small outputs (start near zero to begin like fixed update)
        self._initialize_small()
    
    def _initialize_small(self):
        """Initialize to produce near-zero outputs (like initial fixed update)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z1_pred, z2_pred, error_z1, error_z2, frame_current, frame_next):
        """
        Args:
            z1_pred: Current z1 latent prediction (B, z1_channels, H1, W1)
            z2_pred: Current z2 latent prediction (B, z2_channels, H2, W2)
            error_z1: Error signal for z1 (B, z1_channels, H1, W1)
            error_z2: Error signal for z2 (B, z2_channels, H2, W2)
            frame_current: Current frame (B, 1, H, W)
            frame_next: Next frame (B, 1, H, W)
        
        Returns:
            z1_update: Predicted update for z1 (B, z1_channels, H1, W1)
            z2_update: Predicted update for z2 (B, z2_channels, H2, W2)
        """
        # Spatial size matching
        z1_h, z1_w = z1_pred.shape[2], z1_pred.shape[3]
        
        # Upsample z2 to z1 spatial size
        z2_upsampled = nn.functional.interpolate(
            z2_pred, size=(z1_h, z1_w), mode='bilinear', align_corners=False
        )
        error_z2_upsampled = nn.functional.interpolate(
            error_z2, size=(z1_h, z1_w), mode='bilinear', align_corners=False
        )
        
        # Resize frames to z1 spatial size
        frame_current_resized = nn.functional.interpolate(
            frame_current, size=(z1_h, z1_w), mode='bilinear', align_corners=False
        )
        frame_next_resized = nn.functional.interpolate(
            frame_next, size=(z1_h, z1_w), mode='bilinear', align_corners=False
        )
        
        # Concatenate all context at z1 spatial size
        context = torch.cat([
            z1_pred,
            z2_upsampled,
            error_z1,
            error_z2_upsampled,
            frame_current_resized,
            frame_next_resized
        ], dim=1)
        
        # Shared feature extraction
        shared_features = self.shared_body(context)
        
        # Predict z1 update (at z1 spatial size)
        z1_update = self.z1_update_head(shared_features)
        
        # Predict z2 update (at z1 spatial size, then downsample)
        z2_update_at_z1 = self.z2_update_head(shared_features)
        z2_update = self.downsample_z2(z2_update_at_z1)
        
        return z1_update, z2_update


class LearnedStepSizePredictor(nn.Module):
    """
    Simpler learned predictor: learns step size (alpha) based on context.
    Maintains the gradient descent direction but adapts step size per-channel.
    """
    
    def __init__(self, z1_channels=128, z2_channels=128, hidden_channels=32):
        super().__init__()
        
        # Input: error_z1, error_z2 (we only need errors to determine step size)
        input_channels = z1_channels + z2_channels
        
        # Simple network to predict step sizes
        self.step_predictor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),  # 2 outputs: alpha_z1, alpha_z2
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Initialize to ~0.5 (like original alpha=0.5, though we've reduced to 0.1)
        self._initialize_half()
    
    def _initialize_half(self):
        """Initialize to produce ~0.5 step sizes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.bias)
                # Initialize to output ~0.5 (sigmoid(0) = 0.5)
                nn.init.zeros_(m.weight)
    
    def forward(self, error_z1, error_z2):
        """
        Args:
            error_z1: Error signal for z1 (B, z1_channels, H1, W1)
            error_z2: Error signal for z2 (B, z2_channels, H2, W2)
        
        Returns:
            alpha_z1: Per-channel step size for z1 (B, 1, H1, W1) * scalar
            alpha_z2: Per-channel step size for z2 (B, 1, H2, W2) * scalar
        """
        # Spatial size matching
        z1_h, z1_w = error_z1.shape[2], error_z1.shape[3]
        
        # Resize error_z2 to z1 spatial size
        error_z2_resized = nn.functional.interpolate(
            error_z2, size=(z1_h, z1_w), mode='bilinear', align_corners=False
        )
        
        # Concatenate errors
        errors = torch.cat([error_z1, error_z2_resized], dim=1)
        
        # Predict step sizes
        step_sizes = self.step_predictor(errors)
        
        # Split into z1 and z2 step sizes
        alpha_z1 = step_sizes[:, 0:1]  # (B, 1, H1, W1)
        alpha_z2 = step_sizes[:, 1:2]  # (B, 1, H1, W1)
        
        # Upsample alpha_z2 back to z2 spatial size if needed
        if z1_h != error_z2.shape[2]:
            alpha_z2 = nn.functional.interpolate(
                alpha_z2, size=(error_z2.shape[2], error_z2.shape[3]),
                mode='bilinear', align_corners=False
            )
        
        return alpha_z1, alpha_z2
