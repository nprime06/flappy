"""Flow matching loss functions."""

import torch
import torch.nn.functional as F


def flow_matching_loss(model, z_target, z_cond, action, aug_level, z_0=None):
    """
    Compute flow matching loss.

    Args:
        model: Flow model that predicts velocity
        z_target: Target latent (B, C, H, W)
        z_cond: Conditioning latents (B, k*C, H, W)
        action: Action indices (B,)
        aug_level: Augmentation level indices (B,)
        z_0: Starting noise. If None, sample from N(0,1).
             For reflow, this is the noise paired with z_target.

    Returns:
        loss: Scalar MSE loss
        info: Dict with additional info for logging
    """
    B = z_target.shape[0]
    device = z_target.device

    # Sample starting point (noise)
    if z_0 is None:
        z_0 = torch.randn_like(z_target)

    # Sample timestep
    t = torch.rand(B, device=device)

    # Linear interpolation: z_t = (1-t)*z_0 + t*z_target
    t_expand = t.view(B, 1, 1, 1)
    z_t = (1 - t_expand) * z_0 + t_expand * z_target

    # Target velocity is constant along the linear path
    v_target = z_target - z_0

    # Predict velocity
    v_pred = model(z_t, t, c=action, z_cond=z_cond, aug_level=aug_level)

    # MSE loss
    loss = F.mse_loss(v_pred, v_target)

    info = {
        "v_pred_norm": v_pred.detach().norm(dim=(1, 2, 3)).mean().item(),
        "v_target_norm": v_target.detach().norm(dim=(1, 2, 3)).mean().item(),
    }

    return loss, info
