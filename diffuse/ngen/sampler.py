import torch

@torch.inference_mode()
def euler_sample(model, z_0, z_cond, actions, aug_level, cfg_scale=1.5, num_steps=50, clamp_range=4.0, num_classes=2):
    B = z_0.shape[0]
    device = z_0.device
    dt = 1.0 / num_steps

    z_cond_null = torch.zeros_like(z_cond) # null conditioning
    actions_null = torch.full_like(actions, num_classes) # null token (num_classes = 2)

    z = z_0
    for i in range(num_steps):
        t = torch.full((B,), (i + 0.5) * dt, device=device) # midpoint

        v_cond, _ = model(z, t, z_cond=z_cond, c=actions, aug_level=aug_level)
        v_uncond, _ = model(z, t, z_cond=z_cond_null, c=actions_null, aug_level=aug_level)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        z = z + v * dt
        z = torch.clamp(z, -clamp_range, clamp_range)
    return z

@torch.inference_mode()
def euler_sample_backward(model, z_1, z_cond, actions, aug_level, cfg_scale=1.5, num_steps=50, clamp_range=4.0, num_classes=2):
    B = z_1.shape[0]
    device = z_1.device
    dt = 1.0 / num_steps

    z_cond_null = torch.zeros_like(z_cond) # null conditioning
    actions_null = torch.full_like(actions, num_classes) # null token (num_classes = 2)

    z = z_1
    for i in range(num_steps):
        t = torch.full((B,), 1.0 - (i + 0.5) * dt, device=device) # midpoint

        v_cond, _ = model(z, t, z_cond=z_cond, c=actions, aug_level=aug_level)
        v_uncond, _ = model(z, t, z_cond=z_cond_null, c=actions_null, aug_level=aug_level)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        z = z - v * dt
        z = torch.clamp(z, -clamp_range, clamp_range)
    return z


class ReflowPairGenerator:
    def __init__(self, model, num_steps=50, cfg_scale=1.5, num_classes=2):
        self.model = model
        self.cfg_scale = cfg_scale
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.model.eval()

    def generate(self, z_1, z_cond, actions, aug_level):
        return euler_sample_backward(
            self.model, z_1, z_cond, actions, aug_level,
            cfg_scale=self.cfg_scale, num_steps=self.num_steps,
            num_classes=self.num_classes
        )
