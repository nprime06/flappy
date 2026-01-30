import torch
import torch.nn.functional as F


def flow_matching_loss(model, z_target, z_cond, actions, aug_level, z_0=None,
                       done_labels=None, done_loss_weight=0.1, 
                       action_weight=1.0, null_action_weight=1.0, done_pos_weight=1.0):
    # action_weight: for action=1 samples to fix class imbalance
    # done_pos_weight: for done=1 samples to fix class imbalance (in done BCE loss)

    B = z_target.shape[0]
    device = z_target.device
    if z_0 is None:
        z_0 = torch.randn_like(z_target)

    t = torch.rand(B, device=device).view(B, 1, 1, 1)
    z_t = (1 - t) * z_0 + t * z_target
    v_target = z_target - z_0
    v_pred, done_logit = model(z_t, t, z_cond=z_cond, c=actions, aug_level=aug_level)
    mse_per_sample = ((v_pred - v_target) ** 2).mean(dim=[1, 2, 3])  # (B,)

    target_action = actions[:, -1]  # (B,), target action
    weights = torch.where(target_action == 1, action_weight, 1.0)
    flow_loss = (mse_per_sample * weights).mean()

    done_loss = F.binary_cross_entropy_with_logits(
        done_logit.squeeze(-1), done_labels, pos_weight=torch.tensor([done_pos_weight], device=done_logit.device)
    )

    total_loss = flow_loss + done_loss_weight * done_loss
    return total_loss, flow_loss, done_loss
