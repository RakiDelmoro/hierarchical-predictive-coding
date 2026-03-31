import torch
from core.learned_predictor import LearnedUpdatePredictor


def run_inference_loop_with_adaptive_stopping(model, frame_current, frame_next, 
                                            latent1, latent2, latent1_pred, latent2_pred,
                                            max_steps=5, min_steps=1, clip_value=1.0,
                                            learned_predictor=None, convergence_threshold=1e-4):
    """
    Inference loop with adaptive stopping based on update convergence.
    
    Args:
        model: Hierarchy model
        frame_current: Current frame (B, 1, H, W)
        frame_next: Next frame (B, 1, H, W)
        latent1: Initial z1 latent (B, z1_channels, H1, W1)
        latent2: Initial z2 latent (B, z2_channels, H2, W2)
        latent1_pred: Initial z1 prediction (B, z1_channels, H1, W1)
        latent2_pred: Initial z2 prediction (B, z2_channels, H2, W2)
        max_steps: Maximum number of inference steps
        min_steps: Minimum number of inference steps
        clip_value: Clipping value for updates
        learned_predictor: Learned predictor module (required)
        convergence_threshold: Threshold for stopping (max update magnitude)
    
    Returns:
        Updated latent1_pred, latent2_pred, and all_errors
    """
    assert learned_predictor is not None, "learned_predictor is required"
    
    all_errors = []
    
    for step in range(max_steps):
        # Compute frame prediction and errors
        frame_prediction = model.decode(latent1_pred)
        error_frame = frame_next - frame_prediction

        # Compute latent errors
        latent1_from_latent2 = model.upsample_latent2_to_latent1(latent2_pred)
        latent1_hat = model.latent1_predictor(latent1_from_latent2)
        error_latent1 = latent1_pred - latent1_hat.detach()

        latent2_hat = model.latent2_predictor(model.latent2_transition(latent2))
        error_latent2 = latent2_pred - latent2_hat.detach()

        # Use learned predictor for updates
        latent1_update, latent2_update = learned_predictor(
            latent1_pred, latent2_pred,
            error_latent1, error_latent2,
            frame_current, frame_next
        )

        # Apply clipping
        latent1_update = torch.clamp(latent1_update, -clip_value, clip_value)
        latent2_update = torch.clamp(latent2_update, -clip_value, clip_value)

        # Check for convergence (only after minimum steps)
        update_magnitude = torch.max(torch.abs(latent1_update)) + torch.max(torch.abs(latent2_update))
        
        # Update latents
        latent1_pred = latent1_pred + latent1_update
        latent2_pred = latent2_pred + latent2_update

        all_errors.append((error_frame, error_latent1, error_latent2))
        
        # Adaptive stopping: check if updates are small enough and we've done minimum steps
        if step >= min_steps - 1 and update_magnitude < convergence_threshold:
            break

    return latent1_pred, latent2_pred, all_errors
