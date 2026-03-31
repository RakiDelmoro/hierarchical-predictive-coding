import torch


def compute_energy(error_frame, error_latent1, error_latent2, 
                   z1=None, z2=None, weight_frame=1.0, weight_latent=0.5, lambda_reg=0.1):
    energy_frame = (error_frame ** 2).mean()
    energy_latent1 = (error_latent1 ** 2).mean()
    energy_latent2 = (error_latent2 ** 2).mean()
    energy_total = weight_frame * energy_frame + weight_latent * (energy_latent1 + energy_latent2)
    
    if z1 is not None and z2 is not None:
        latent_reg = (z1 ** 2).mean() + (z2 ** 2).mean()
        energy_total = energy_total + lambda_reg * latent_reg
    
    return energy_total, energy_frame, energy_latent1, energy_latent2
