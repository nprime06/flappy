"""ODE sampling and reflow utilities for flow matching."""

import torch


@torch.no_grad()
def euler_sample(model, z_0, z_cond, action, aug_level, num_steps=50):
    """
    Sample from flow model using Euler ODE solver (forward: t=0 → t=1).

    Integrates: dz/dt = v(z, t) from t=0 to t=1

    Args:
        model: Flow model predicting velocity
        z_0: Starting noise (B, C, H, W)
        z_cond: Conditioning latents (B, k*C, H, W)
        action: Action indices (B,)
        aug_level: Augmentation level indices (B,)
        num_steps: Number of Euler steps

    Returns:
        z_1: Final sample at t=1 (B, C, H, W)
    """
    B = z_0.shape[0]
    device = z_0.device
    dt = 1.0 / num_steps

    z = z_0
    for i in range(num_steps):
        t = torch.full((B,), (i + 0.5) * dt, device=device)  # Midpoint rule
        v = model(z, t, c=action, z_cond=z_cond, aug_level=aug_level)
        z = z + v * dt

    return z


@torch.no_grad()
def euler_sample_backward(model, z_1, z_cond, action, aug_level, num_steps=50):
    """
    Infer z_0 from z_1 by integrating ODE backward (t=1 → t=0).

    Used for reflow: given real data z_1, find the noise z_0 that maps to it.

    Args:
        model: Flow model predicting velocity
        z_1: Data point at t=1 (B, C, H, W)
        z_cond: Conditioning latents (B, k*C, H, W)
        action: Action indices (B,)
        aug_level: Augmentation level indices (B,)
        num_steps: Number of Euler steps

    Returns:
        z_0: Inferred noise at t=0 (B, C, H, W)
    """
    B = z_1.shape[0]
    device = z_1.device
    dt = 1.0 / num_steps

    z = z_1
    for i in range(num_steps):
        t = torch.full((B,), 1.0 - (i + 0.5) * dt, device=device)  # Midpoint rule
        v = model(z, t, c=action, z_cond=z_cond, aug_level=aug_level)
        z = z - v * dt  # subtract (backward direction)

    return z


@torch.no_grad()
def euler_sample_cfg(model, z_0, z_cond, action, aug_level, cfg_scale=1.5, num_steps=50):
    """
    Sample from flow model using Euler ODE solver with Classifier-Free Guidance.

    Integrates: dz/dt = v(z, t) from t=0 to t=1 with CFG interpolation.

    Args:
        model: Flow model predicting velocity
        z_0: Starting noise (B, C, H, W)
        z_cond: Conditioning latents (B, k*C, H, W)
        action: Action indices (B,)
        aug_level: Augmentation level indices (B,)
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
        num_steps: Number of Euler steps

    Returns:
        z_1: Final sample at t=1 (B, C, H, W)
    """
    B = z_0.shape[0]
    device = z_0.device
    dt = 1.0 / num_steps
    z_cond_null = torch.zeros_like(z_cond)

    z = z_0
    for i in range(num_steps):
        t = torch.full((B,), (i + 0.5) * dt, device=device)  # Midpoint rule

        v_cond = model(z, t, c=action, z_cond=z_cond, aug_level=aug_level)
        v_uncond = model(z, t, c=action, z_cond=z_cond_null, aug_level=aug_level)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        z = z + v * dt

    return z


class ReflowPairGenerator:
    """
    Generate z_0 for reflow training by flowing backward from real data.

    In reflow:
    1. z_1 = real data (from dataset)
    2. z_0 = flow_backward(z_1) using pre-trained model
    3. Train new model on (z_0, z_1) pairs

    This keeps z_1 grounded in real data while straightening the flow paths.

    Usage:
        generator = ReflowPairGenerator(pretrained_model, num_steps=50)

        # In training loop:
        z_0 = generator.generate(z_target, z_cond, action, aug_level)
        loss = flow_matching_loss(model, z_target, z_cond, action, aug_level, z_0=z_0)
    """

    def __init__(self, model, num_steps=50):
        """
        Args:
            model: Pre-trained flow model for inferring z_0
            num_steps: Number of Euler steps for backward integration
        """
        self.model = model
        self.num_steps = num_steps
        self.model.eval()

    def generate(self, z_1, z_cond, action, aug_level):
        """
        Infer z_0 from real data z_1 by flowing backward.

        Args:
            z_1: Real data latent (B, C, H, W)
            z_cond: Conditioning latents (B, k*C, H, W)
            action: Action indices (B,)
            aug_level: Augmentation level indices (B,)

        Returns:
            z_0: Inferred noise (B, C, H, W)
        """
        return euler_sample_backward(
            self.model, z_1, z_cond, action, aug_level,
            num_steps=self.num_steps
        )
