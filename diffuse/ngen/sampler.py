import torch


def _unwrap_model(model):
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def _model_num_classes(model, fallback=2):
    model = _unwrap_model(model)
    class_embedding = getattr(model, "class_embedding", None)
    if class_embedding is not None:
        return int(class_embedding.num_embeddings)
    return int(fallback)


def _cfg_inputs(model, z_cond, actions, cfg_mode, null_action_id, num_classes):
    if cfg_mode == "none":
        return z_cond, actions

    inferred_num_classes = _model_num_classes(model, fallback=num_classes)
    if cfg_mode == "auto":
        cfg_mode = "action" if inferred_num_classes > 2 else "frame"

    if cfg_mode == "action":
        if null_action_id is None:
            null_action_id = inferred_num_classes - 1
        if null_action_id < 0 or null_action_id >= inferred_num_classes:
            raise ValueError(f"null_action_id={null_action_id} outside num_classes={inferred_num_classes}")
        return z_cond, torch.full_like(actions, int(null_action_id))

    if cfg_mode == "frame":
        return torch.zeros_like(z_cond), actions

    raise ValueError(f"unknown cfg_mode: {cfg_mode}")


@torch.inference_mode()
def euler_sample(
    model,
    z_0,
    z_cond,
    actions,
    aug_level,
    cfg_scale=1.5,
    num_steps=50,
    clamp_range=4.0,
    num_classes=2,
    cfg_mode="auto",
    null_action_id=None,
):
    B = z_0.shape[0]
    device = z_0.device
    dt = 1.0 / num_steps

    z = z_0
    for i in range(num_steps):
        t = torch.full((B,), (i + 0.5) * dt, device=device) # midpoint

        v_cond, *_ = model(z, t, z_cond=z_cond, c=actions, aug_level=aug_level)
        if cfg_scale == 1.0 or cfg_mode == "none":
            v = v_cond
        else:
            z_cond_null, actions_null = _cfg_inputs(
                model,
                z_cond,
                actions,
                cfg_mode=cfg_mode,
                null_action_id=null_action_id,
                num_classes=num_classes,
            )
            v_uncond, *_ = model(z, t, z_cond=z_cond_null, c=actions_null, aug_level=aug_level)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

        z = z + v * dt
        z = torch.clamp(z, -clamp_range, clamp_range)
    return z

@torch.inference_mode()
def euler_sample_backward(
    model,
    z_1,
    z_cond,
    actions,
    aug_level,
    cfg_scale=1.5,
    num_steps=50,
    clamp_range=4.0,
    num_classes=2,
    cfg_mode="auto",
    null_action_id=None,
):
    B = z_1.shape[0]
    device = z_1.device
    dt = 1.0 / num_steps

    z = z_1
    for i in range(num_steps):
        t = torch.full((B,), 1.0 - (i + 0.5) * dt, device=device) # midpoint

        v_cond, *_ = model(z, t, z_cond=z_cond, c=actions, aug_level=aug_level)
        if cfg_scale == 1.0 or cfg_mode == "none":
            v = v_cond
        else:
            z_cond_null, actions_null = _cfg_inputs(
                model,
                z_cond,
                actions,
                cfg_mode=cfg_mode,
                null_action_id=null_action_id,
                num_classes=num_classes,
            )
            v_uncond, *_ = model(z, t, z_cond=z_cond_null, c=actions_null, aug_level=aug_level)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

        z = z - v * dt
        z = torch.clamp(z, -clamp_range, clamp_range)
    return z


class ReflowPairGenerator:
    def __init__(self, model, num_steps=50, cfg_scale=1.5, num_classes=2, cfg_mode="auto", null_action_id=None):
        self.model = model
        self.cfg_scale = cfg_scale
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.cfg_mode = cfg_mode
        self.null_action_id = null_action_id
        self.model.eval()

    def generate(self, z_1, z_cond, actions, aug_level):
        return euler_sample_backward(
            self.model, z_1, z_cond, actions, aug_level,
            cfg_scale=self.cfg_scale, num_steps=self.num_steps,
            num_classes=self.num_classes, cfg_mode=self.cfg_mode,
            null_action_id=self.null_action_id,
        )
