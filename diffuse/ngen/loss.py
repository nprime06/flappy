import torch
import torch.nn.functional as F


def flow_matching_loss(model, z_target, z_cond, actions, aug_level, z_0=None,
                       done_labels=None, done_loss_weight=0.1,
                       action_weight=1.0, null_action_weight=1.0, done_pos_weight=1.0,
                       done_t_power=0.0,
                       game_states=None, dynamics_loss_weight=0.0):
    # action_weight: for action=1 samples to fix class imbalance
    # done_pos_weight: for done=1 samples to fix class imbalance (in done BCE loss)
    # done_t_power: reweight done loss by normalized w(t) propto t^alpha

    B = z_target.shape[0]
    device = z_target.device
    if z_0 is None:
        z_0 = torch.randn_like(z_target)

    t = torch.rand(B, device=device)
    z_t = (1 - t.view(B, 1, 1, 1)) * z_0 + t.view(B, 1, 1, 1) * z_target
    v_target = z_target - z_0
    v_pred, done_logit, dynamics_pred = model(z_t, t, z_cond=z_cond, c=actions, aug_level=aug_level)
    mse_per_sample = ((v_pred - v_target) ** 2).mean(dim=[1, 2, 3])  # (B,)

    target_action = actions[:, -1]  # (B,), target action
    weights = torch.where(target_action == 1, action_weight, 1.0)
    weights = weights.to(dtype=mse_per_sample.dtype) # weighted mean for stable per-batch magnitude
    flow_loss = (mse_per_sample * weights).sum() / weights.sum().clamp_min(1e-12)

    pos_weight = torch.tensor(done_pos_weight, device=done_logit.device, dtype=done_logit.dtype)
    done_bce_per_sample = F.binary_cross_entropy_with_logits(
        done_logit.squeeze(-1),
        done_labels,
        pos_weight=pos_weight,
        reduction="none",
    )  # (B,)
    # weight done loss by normalized w(t) ∝ t^alpha.
    w_norm = (done_t_power + 1.0) * torch.pow(t, done_t_power)  # mean-1 under t~U(0,1)
    w_norm = w_norm.to(dtype=done_bce_per_sample.dtype) # weighted mean for stable per-batch magnitude
    done_loss = (done_bce_per_sample * w_norm).sum() / w_norm.sum().clamp_min(1e-12)

    # auxiliary dynamics loss: predict game state from bottleneck (not detached)
    dynamics_loss = torch.tensor(0.0, device=device)
    if dynamics_pred is not None and game_states is not None and dynamics_loss_weight > 0:
        dynamics_loss = F.mse_loss(dynamics_pred, game_states)

    total_loss = flow_loss + done_loss_weight * done_loss + dynamics_loss_weight * dynamics_loss
    return total_loss, flow_loss, done_loss, dynamics_loss
