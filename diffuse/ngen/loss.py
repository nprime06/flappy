"""Flow matching loss functions."""

import torch
import torch.nn.functional as F


def flow_matching_loss(model, z_target, z_cond, action, aug_level, z_0=None,
                       done_labels=None, done_loss_weight=0.1, action_weight=1.0):
    """
    Compute flow matching loss with optional done head loss.

    Args:
        model: Flow model that predicts velocity
        z_target: Target latent (B, C, H, W)
        z_cond: Conditioning latents (B, k*C, H, W)
        action: Action indices (B,)
        aug_level: Augmentation level indices (B,)
        z_0: Starting noise. If None, sample from N(0,1).
             For reflow, this is the noise paired with z_target.
        done_labels: Binary labels for termination (B,). If None, skip done loss.
        done_loss_weight: Weight for done head BCE loss.
        action_weight: Loss weight multiplier for action=1 samples (to fix class imbalance).

    Returns:
        loss: Total loss (flow + done)
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

    # Predict velocity (and optionally done logit)
    if done_labels is not None:
        v_pred, done_logit = model(z_t, t, c=action, z_cond=z_cond,
                                    aug_level=aug_level, return_done=True)
    else:
        v_pred = model(z_t, t, c=action, z_cond=z_cond, aug_level=aug_level)

    # MSE loss for flow matching (per-sample, then weighted)
    mse_per_sample = ((v_pred - v_target) ** 2).mean(dim=[1, 2, 3])  # (B,)

    # Apply higher weight to action=1 samples to fix class imbalance
    if action_weight > 1.0:
        weights = torch.where(action == 1, action_weight, 1.0)
        flow_loss = (mse_per_sample * weights).mean()
    else:
        flow_loss = mse_per_sample.mean()

    info = {
        "v_pred_norm": v_pred.detach().flatten(1).norm(dim=1).mean().item(),
        "v_target_norm": v_target.detach().flatten(1).norm(dim=1).mean().item(),
    }

    # Add done loss if labels provided
    if done_labels is not None:
        done_loss = F.binary_cross_entropy_with_logits(done_logit.squeeze(-1), done_labels)
        total_loss = flow_loss + done_loss_weight * done_loss
        info["done_loss"] = done_loss.item()
    else:
        total_loss = flow_loss

    return total_loss, info
