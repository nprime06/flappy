"""ODE sampling and reflow utilities for flow matching."""

import torch


@torch.no_grad()
def euler_sample(model, z_0, z_cond, action, aug_level, num_steps=50):
    """
    Sample from flow model using Euler ODE solver.

    Integrates: dz/dt = v(z, t) from t=0 to t=1

    Args:
        model: Flow model predicting velocity
        z_0: Starting noise (B, C, H, W)
        z_cond: Conditioning latents (B, k*C, H, W)
        action: Action indices (B,)
        aug_level: Augmentation level indices (B,)
        num_steps: Number of Euler steps

    Returns:
        z_1: Final sample (B, C, H, W)
    """
    B = z_0.shape[0]
    device = z_0.device
    dt = 1.0 / num_steps

    z = z_0
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model(z, t, c=action, z_cond=z_cond, aug_level=aug_level)
        z = z + v * dt

    return z


@torch.no_grad()
def generate_reflow_pairs(model, z_target_shape, z_cond, action, aug_level, num_steps=50):
    """
    Generate (z_0, z_1) pairs for rectified flow reflow training.

    In reflow:
    1. Sample z_0 ~ N(0,1)
    2. Flow forward using the trained model to get z_1
    3. Use (z_0, z_1) as training pairs for the next model

    This "straightens" the learned flow paths, enabling fewer sampling steps.

    Args:
        model: Pre-trained flow model
        z_target_shape: Shape of latents (B, C, H, W)
        z_cond: Conditioning latents (B, k*C, H, W)
        action: Action indices (B,)
        aug_level: Augmentation level indices (B,)
        num_steps: Number of Euler steps for generation

    Returns:
        z_0: Starting noise (B, C, H, W)
        z_1: Generated endpoints (B, C, H, W)
    """
    device = z_cond.device

    # Sample starting noise
    z_0 = torch.randn(z_target_shape, device=device)

    # Flow forward to generate z_1
    z_1 = euler_sample(model, z_0, z_cond, action, aug_level, num_steps=num_steps)

    return z_0, z_1


class ReflowPairGenerator:
    """
    Wrapper for generating reflow pairs during training.

    Usage:
        generator = ReflowPairGenerator(pretrained_model, num_steps=50)

        # In training loop:
        z_0, z_1 = generator.generate(z_target_shape, z_cond, action, aug_level)
        loss = flow_matching_loss(model, z_1, z_cond, action, aug_level, z_0=z_0)
    """

    def __init__(self, model, num_steps=50):
        """
        Args:
            model: Pre-trained flow model for generating pairs
            num_steps: Number of Euler steps
        """
        self.model = model
        self.num_steps = num_steps
        self.model.eval()

    def generate(self, z_target_shape, z_cond, action, aug_level):
        """Generate (z_0, z_1) reflow pairs."""
        return generate_reflow_pairs(
            self.model, z_target_shape, z_cond, action, aug_level,
            num_steps=self.num_steps
        )
