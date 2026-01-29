BHL# Flappy World Model - Architecture Reference

A neural world model that learns to simulate Flappy Bird by predicting next frames from past frames + actions.

```
PPO Agent ──► VOD Recording ──► VAE Training ──► Latent Encoding ──► Flow Model ──► Deployment
   │              │                  │                 │                  │              │
   │              ▼                  ▼                 ▼                  ▼              ▼
   │         vod/p_stim_*/      diffuse/vae/       latent-vod/       diffuse/ngen/    web/ or
   ▼         {frames/,          runs/vae_*/       {latents.pt,      runs/ngen_*/    world/
game/rl/     run_info.jsonl}    config.json       encode_config}    checkpoints/
runs/
```

---

## Stage 1: Game Environment (`game/`)

### What it does
Provides the Flappy Bird gym environment and trains a PPO agent to play it.

### Key design choices

**Observation reduction** (`environment.py:127-137`):
```python
# Raw obs is 12-dim: pipe positions, bird state, etc.
# Reduced to 3-dim for PPO:
dy = (gap_middle_y - bird_y - 0.01) / 0.11  # distance to gap center
bird_vel_y                                    # vertical velocity
dx = (nearest_pipe_x - 0.1) / 0.3            # distance to next pipe
```
The magic constants (0.01, 0.11, 0.1, 0.3) normalize these to roughly [-1, 1].

**Death frames**: `DEATH_FRAMES = 5` extra frames are captured after termination with game-over overlay. These teach the flow model what "dying" looks like.

**run_info.jsonl format**: Streams per-step data including:
- `action`: 0 or 1
- `terminated`: used as `done` label for flow model
- `processed_obs`: not used downstream (actions are sufficient)

### Output consumed by
`vod/record.py` loads the PPO checkpoint to record gameplay.

---

## Stage 2: VOD Recording (`vod/record.py`)

### What it does
Records gameplay videos with behavioral diversity by randomly overriding the PPO agent's decisions.

### Key design choices

**HijackedPPOAgent**: Wraps the trained agent with:
- `p_stim`: probability of forcing flap when model says don't
- `p_freeze`: probability of blocking flap when model says flap

This creates diverse training data (different pipe-bird configurations) beyond what the optimal policy would generate.

**Directory structure**: `vod/p_stim_X_p_freeze_Y/TIMESTAMP/`
- `frames/000000.png, 000001.png, ...` - 288x512 RGB
- `run_info.jsonl` - actions + terminated flags per step
- `metrics.json` - episode summary

**Frame-step alignment**: Frame `N` shows the state AFTER action `N` was executed. So action at step `N` causes frame `N`.

### Output consumed by
`latent-vod/encode_vod.py` reads frames + run_info.jsonl.

---

## Stage 3: VAE Training (`diffuse/vae/`)

### What it does
Trains an autoencoder to compress 288x512 frames to 4x32x18 latent tensors.

### Key design choices

**Bird detection for weighted loss** (`train_vae.py:63-98`):
```python
# Detect orange pixels (bird body)
bird_mask = (r > 0.8) & (g > 0.3) & (g < 0.6) & (b < 0.3)
# Find top of bird, offset by 23px to capture white head
y_min = rows.min() - 23
# Fixed x range because bird doesn't move horizontally (world scrolls)
weights[..., y_min:y_max, 50:100] = 10.0  # 10x weight on bird region
```

**Loss composition**:
- Weighted L1 (10x on bird) - reconstructs bird accurately
- Gradient L1 - preserves edges
- KL divergence (weight=0.001) - regularizes latent space

**Latent statistics computed at end** (`train_vae.py:106-151`):
After training, samples 500 random frames, encodes them, and saves `latent_mean` and `latent_std` to `config.json`. These are used by encode_vod.py for normalization.

**Spatial dimensions**: 288x512 → 4 downsamples (2^4=16) → 18x32. But width is 288/16=18, height is 512/16=32, so latent is 4x32x18 (C, H, W).

### Output consumed by
`latent-vod/encode_vod.py` loads checkpoint + `config.json` for latent statistics.

---

## Stage 4: Latent Encoding (`latent-vod/encode_vod.py`)

### What it does
Batch-encodes all VOD frames to normalized latent tensors.

### Key design choices

**Normalization applied here, not in VAE** (`encode_vod.py:63-67`):
```python
def encode(vae, x, latent_mean, latent_std):
    z, _, _ = vae.reparameterize(encoded, sample=False)  # use mean, not sample
    z = (z - latent_mean) / latent_std  # <-- normalization happens here
    return z
```
This keeps VAE training standard while normalizing for flow model training.

**Copies run_info.jsonl** alongside `latents.pt` because actions are needed for flow training.

**Computes class weights** (`compute_dataset_statistics`):
- `action_weight = count(action=0) / count(action=1)` ~17x (flaps are rare)
- `done_pos_weight = count(done=0) / count(done=1)` ~30-159x (deaths are rare)

These weights are saved to `encode_config.json` and loaded by flow training.

### Output consumed by
`diffuse/ngen/ngen_data.py` loads latents.pt + run_info.jsonl.
`diffuse/ngen/train_ngen.py` loads encode_config.json for class weights.

---

## Stage 5: Flow Model (`diffuse/ngen/`)

### What it does
Trains a flow matching model to predict next-frame latents given k past latents + k+1 actions.

### Key design choices

**Context window** (`ngen_data.py`):
- k past latents: `latents[step-k:step]` → shape (k, 4, 32, 18)
- k+1 actions: `[a_{step-k}, ..., a_{step-1}, a_{step}]`
- The final action `a_{step}` is the one that causes the target frame `latents[step]`

**ResUNet conditioning** (`resunet.py`):
```python
x = torch.cat([x, z_cond], dim=1)  # current + k past → (k+1)*4 channels
# All conditioning (time, actions, aug) via AdaGN:
t_emb = TimeEmbedding(t)           # sinusoidal
c_emb = Linear(flatten(action_embeddings))
aug_emb = Embedding(aug_level)
# In each ResBlock: x = x * (scale + 1) + shift
```

**Flow matching loss** (`loss.py:24-38`):
```python
t = torch.rand(B)  # sample time uniformly
z_t = (1 - t) * z_0 + t * z_target  # linear interpolation
v_target = z_target - z_0           # constant velocity field
loss = MSE(v_pred, v_target)
```
This is simpler than denoising diffusion - just learn a straight path.

**Action weighting** (`loss.py:33-38`):
```python
weights = torch.where(target_action == 1, action_weight, 1.0)
flow_loss = (mse_per_sample * weights).mean()
```
Upweights flap samples ~17x to handle class imbalance.

**Done head**: Auxiliary task predicting termination from bottleneck features:
```python
done_logit = AdaptiveAvgPool2d(1) → Linear(hidden, 64) → Linear(64, 1)
done_loss = BCE(done_logit, done_labels, pos_weight=done_pos_weight)
total_loss = flow_loss + 0.1 * done_loss
```

**Noise augmentation** (`train_ngen.py:302-304`):
```python
aug_level = randint(0, 16)          # 16 bins
aug_std = aug_level / 16 * 0.5      # maps to [0, 0.5]
z_cond = z_cond + randn * aug_std   # add noise to conditioning
```
Makes model robust to noisy observations during inference.

**CFG dropout** (`train_ngen.py:306-312`):
10% of training samples replace conditioning with zeros and actions with null token (index 2). Enables classifier-free guidance at inference.

**ODE midpoint rule** (`sampler.py:11`):
```python
t = (i + 0.5) * dt  # NOT i * dt
```
Evaluates velocity at midpoint of each step for better accuracy.

**Latent clipping** (`sampler.py:14`):
```python
z = torch.clamp(z, -clamp_range, clamp_range)  # default [-4, 4]
```
Prevents latent drift during autoregressive generation.

**Reflow** (`sampler.py:121-177`):
Two-stage training:
1. Train initial model on `z_0 ~ N(0,1)` → `z_1 = data`
2. Train reflow model on `z_0 = backward(z_1)` → `z_1 = data`
This straightens flow paths for fewer integration steps at inference.

### Output consumed by
`diffuse/export/export_onnx.py` and `world/test_world.py` load checkpoints.

---

## Stage 6: Deployment

### Python (`world/test_world.py`)
Interactive pygame loop:
1. Maintain sliding window of k past latents
2. On each frame:
   - `z_0 = current_latent + noise * scale` (perturbation, not pure noise)
   - `z_next = euler_sample(z_0, z_cond, actions)`
   - Decode with VAE, display
   - Check done_prob from done head

### Browser (`web/`)
Uses ONNX models via onnxruntime-web:
- `export_onnx.py` wraps models to avoid graph issues (`randn_like` doesn't export cleanly)
- `EulerSampler.ts` implements Euler integration in TypeScript
- `FrameBuffer.ts` manages the sliding context window

---

## File Dependency Graph

```
game/rl/runs/*/checkpoints/latest.pt (PPO)
    │
    ▼ record.py reads
vod/p_stim_*/TIMESTAMP/
    ├── frames/*.png
    └── run_info.jsonl
    │
    ▼ encode_vod.py reads (+ VAE checkpoint)
diffuse/vae/runs/vae_*/
    ├── checkpoints/latest.pt
    └── config.json (latent_mean, latent_std)
    │
    ▼ encode_vod.py writes
latent-vod/
    ├── p_stim_*/TIMESTAMP/
    │   ├── latents.pt
    │   └── run_info.jsonl
    └── encode_config.json (action_weight, done_pos_weight)
    │
    ▼ train_ngen.py reads
diffuse/ngen/runs/ngen_*/
    └── checkpoints/latest.pt
    │
    ├─▼ export_onnx.py reads
    │  web/public/models/*.onnx
    │
    └─▼ test_world.py reads
       (interactive inference)
```

---

## Quick Reference

| Constant | Value | Where used |
|----------|-------|------------|
| Image size | 288x512 | VOD frames |
| Latent shape | 4x32x18 | VAE output |
| Latent mean | 0.4755 | encode_config.json |
| Latent std | 1.5959 | encode_config.json |
| Context frames (k) | 8 | flow model |
| Actions per sample | k+1 = 9 | flow model |
| Augmentation bins | 16 | flow model |
| Latent clamp range | [-4, 4] | sampler.py |
| ODE steps (default) | 50 | sampler.py |
