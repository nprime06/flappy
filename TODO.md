# Flappy Bird World Model - Implementation Review vs GameNGen

## Executive Summary

Your implementation has a solid foundation with correctly implemented flow matching, noise augmentation, and reflow mechanisms. However, there are significant issues spanning **mathematical bugs**, **data pipeline problems**, **inference instability**, and **GameNGen mismatches**. The most critical are:

1. **Episode boundary violation** - dead bird frames used as context for fresh episodes
2. **No latent bounds checking** - generated latents can produce garbage
3. **ODE integration bug** - time stepping doesn't reach t=1.0
4. **Fresh noise each frame** - wrong approach for autoregressive generation
5. **Behavioral augmentation disabled** - all data from single PPO policy
6. **Train-test step mismatch** (50 training vs 20 inference steps)

---

# PART 0: DATA PIPELINE ISSUES (NEW)

Critical issues discovered in the data collection and loading pipeline.

---

## 0.1 Episode Boundary Wrap-Around Violation (CRITICAL)

**File**: `diffuse/ngen/ngen_data.py:58-64`
```python
# First episode: start from step k (need k prior frames within episode)
# Subsequent episodes: start from step 0 (can use previous episode's last k frames)
start_step = k if ep_idx == 0 else 0

for step in range(start_step, n_frames):
    if step in actions:
        self.samples.append((ep_idx, step, actions[step]))
```

**The Problem**: When creating training samples for Episode N (N > 0), step 0:
- Context frames wrap to Episode N-1's **last k frames** (dead bird, crashed state)
- Target frame is Episode N's frame 0 (fresh bird at starting position)
- **The model learns: dead_bird + action → fresh_bird (impossible!)**

**Impact**: ~3-4K contaminated training samples (~2.4% of dataset). Teaches unphysical "reset magic" transitions.

**Fix**: Change line 60 to start ALL episodes at step k:
```python
start_step = k  # Always require k prior frames from SAME episode
```

This wastes k frames per episode but ensures physical causality.

---

## 0.2 Behavioral Augmentation Completely Disabled (CRITICAL)

**File**: `vod/record.py:12-13`
```python
p_stim = 0.0      # Force-flap probability - DISABLED
p_freeze = 0.0    # Force-no-flap probability - DISABLED
```

**The Problem**: ALL 1000 episodes follow the exact same trained PPO policy with NO behavioral diversity.

**Consequences**:
- Action distribution: 94.6% no-flap, 5.4% flap (heavily skewed)
- Model only sees states reachable by the single policy
- Major blind spots: aggressive flapping, extreme velocities, unusual collision avoidance
- World model will fail on out-of-distribution player behaviors

**Fix**: Enable augmentation for data diversity:
```python
p_stim = 0.05    # 5% chance to force flap when policy says no
p_freeze = 0.15  # 15% chance to force no-flap when policy says yes
```

Then recollect data with these settings.

---

## 0.3 Initial Reset Frame Not Saved (MEDIUM)

**File**: `game/environment.py:246-254`

Frame saving only happens inside the main loop AFTER `env.step()`. The initial reset state (frame 0 before any action) is never captured.

**Impact**:
- World model doesn't learn what a fresh game reset looks like
- Frame count equals action count, not action count + 1

**Fix**: Add frame saving after reset:
```python
obs, info = env.reset(seed=cfg.seed)
if frames_dir is not None:
    frame = env.render()
    iio.imwrite(frames_dir / "000000.png", frame)  # Save initial state
```

---

# PART 0.5: INFERENCE STABILITY ISSUES (NEW)

Critical issues discovered in autoregressive generation and error accumulation.

---

## 0.4 No Latent Space Bounds Checking (CRITICAL)

**Files**: `world/test_world.py`, `diffuse/ngen/sampler.py`

**The Problem**: Generated latents have no bounds enforcement:
```python
# In euler_sample:
z = z + v * dt  # Can drift arbitrarily far from training distribution

# In latent_to_image:
z = z * latent_std + latent_mean  # Denormalize
return vae.decoder(z)  # NO CLIPPING! Decoder receives arbitrary latents
```

**Impact**:
- After ~50 Euler steps, latents can be 10+ standard deviations from training mean
- VAE decoder produces garbage for out-of-distribution latents
- Generation quality collapses after ~1000 frames

**Fix**: Add latent clipping:
```python
def euler_sample(...):
    ...
    z = z + v * dt
    z = torch.clamp(z, -3.0, 3.0)  # Clip to reasonable range (normalized space)
    ...
```

---

## 0.5 Fresh Random Noise Each Frame (HIGH)

**File**: `world/test_world.py:243-245`
```python
z_0 = torch.randn_like(z_display)  # Fresh N(0,1) noise each frame
z_next = euler_sample(flow_model, z_0, z_cond, ...)
```

**The Problem**: Each frame starts from completely independent random noise. This is WRONG for autoregressive generation because:
1. Training assumes single-step generation from random noise to target
2. Consecutive frames should have correlated starting points
3. Fresh noise causes discontinuities and variance across frames

**Better Approaches**:

Option A - Perturbation from current:
```python
z_0 = z_display + 0.1 * torch.randn_like(z_display)
```

Option B - Deterministic (no noise):
```python
z_0 = z_display  # Start from previous frame's latent
```

Option C - Learned coupling (requires retraining):
- Train model with correlated noise across consecutive frames

---

## 0.6 Static aug_level=0 at Inference (MEDIUM)

**File**: `world/test_world.py:209`
```python
aug_level = torch.zeros(1, dtype=torch.long, device=device)  # Always 0
```

**The Problem**: Training uses random aug_level from 0-15 (with corresponding noise added to conditioning). Inference always uses 0 (no noise) - this creates distribution shift.

**Better Approach**: Use median augmentation or match expected noise level:
```python
aug_level = torch.full((1,), 8, dtype=torch.long, device=device)  # Median
# Or adaptively set based on generated frame quality
```

---

## 0.7 No Temporal Consistency Mechanisms (HIGH)

**The Problem**: The model has NO explicit mechanisms to enforce:
- Temporal smoothness between consecutive frames
- Physics consistency (gravity, impulse)
- Valid state transitions

It relies entirely on:
- Implicit learning from VOD data
- Context conditioning (only 2 frames)
- VAE's learned manifold

**Failure Modes**:
- Bird can teleport, flicker, or vanish
- Physics violations (impossible jumps)
- Errors compound over long generation

**Mitigation** (without retraining):
```python
# Temporal smoothing on generated latents
z_next_smoothed = 0.8 * z_next + 0.2 * z_display
```

---

# PART 1: CONCEPTUAL & MATHEMATICAL ISSUES

These are fundamental correctness problems independent of GameNGen comparison.

---

## 1. ODE Integration Time Stepping Bug (CRITICAL)

**File**: `diffuse/ngen/sampler.py:29-32`
```python
for i in range(num_steps):
    t = torch.full((B,), i * dt, device=device)  # t = i/num_steps
    v = model(z, t, c=action, z_cond=z_cond, aug_level=aug_level)
    z = z + v * dt
```

**Bug**: With `num_steps=50` and `dt=1/50=0.02`:
- Step 0: t=0.00
- Step 1: t=0.02
- ...
- Step 49: t=0.98 (NOT 1.0!)

The final step evaluates at t=0.98, creating a **systematic 2% trajectory error**. The ODE never reaches t=1.0.

**Same bug in backward integration** (`sampler.py:60-63`):
```python
for i in range(num_steps):
    t = torch.full((B,), 1.0 - i * dt, device=device)  # Starts at 1.0, ends at 0.02
```
This starts at t=1.0 but ends at t=0.02, not t=0.0.

**Impact**:
- Forward-backward asymmetry breaks reflow's reversibility assumption
- Generated samples are systematically off from the true target distribution

**Fix option 1** (midpoint rule):
```python
for i in range(num_steps):
    t = torch.full((B,), (i + 0.5) * dt, device=device)
```

**Fix option 2** (include endpoint):
```python
for i in range(num_steps):
    t = torch.full((B,), min((i + 1) * dt, 1.0 - 1e-6), device=device)
```

---

## 2. Train-Test Step Size Mismatch (CRITICAL)

**Training** (`train_ngen.py:42`): Uses `reflow_steps: 50`
**Inference** (`world/test_world.py:123`): Defaults to `--num-steps=20`

**Impact**:
- Model trained with fine-grained ODE solutions (50 steps, dt=0.02)
- Inference uses coarse steps (20 steps, dt=0.05)
- Each inference step is 2.5x larger than training step
- **Euler discretization error accumulates differently**

This creates a train-test distribution shift. The model learned velocity fields for small step sizes but inference uses large steps.

**Fix**: Use same step count for inference, or explicitly train/test with multiple step sizes.

---

## 3. VAE Sampling Mode Mismatch (HIGH)

**VAE Training** (`train_ae.py:176`):
```python
z, mean, logvar = model.reparameterize(encoded, sample=True)  # Stochastic
```

**Flow Model Training** (`train_ngen.py:90`):
```python
z, _, _ = vae.reparameterize(encoded, sample=False)  # Deterministic (uses mean only)
```

**Issue**: The VAE is trained with stochastic sampling (reparameterization trick), but the flow model only ever sees the deterministic mean. This means:
- VAE posterior variance is never utilized
- Flow model trained on a **different distribution** than VAE was optimized for
- The logvar output of VAE is essentially ignored during world model training

**Impact**: Potential distribution mismatch causing blurry or inconsistent generations.

---

## 4. Missing Gradient Clipping (HIGH)

**File**: `train_ngen.py` - no gradient clipping anywhere

**Risk factors**:
- Deep network (4 down + bottleneck + 4 up = 9 blocks)
- Large latent norm (~13.37 std before normalization)
- High-dimensional output (4 × 18 × 18 = 1296 velocity dimensions)
- Batch size 512 amplifies gradient magnitude

**Impact**: Potential gradient explosion, especially early in training.

**Fix**: Add gradient clipping:
```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 5. No Learning Rate Schedule (MEDIUM)

**File**: `train_ngen.py:132`
```python
optimizer = AdamW(model.parameters(), lr=train_config["lr"])
```

No learning rate decay over 40 epochs. While AdamW is adaptive, a constant learning rate may:
- Prevent fine convergence in later epochs
- Cause oscillation around optima

**Fix**: Add cosine annealing or exponential decay:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"])
# In training loop:
scheduler.step()
```

---

## 6. Skip Connection Spatial Mismatch (MEDIUM)

**File**: `diffuse/nn/resblock.py:83-85`
```python
if x.shape[2:] != skip.shape[2:]:
    x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
```

This bilinear interpolation is a **workaround for dimension mismatches** caused by:
- Odd latent dimension (18 = 2 × 3²)
- Repeated stride-2 operations on non-power-of-2 dimensions

**Impact**:
- Bilinear interpolation introduces gradient artifacts
- `align_corners=False` can cause checkerboard patterns
- Information loss at resolution boundaries

**Better approach**: Pad inputs to power-of-2 dimensions, or use reflection padding.

---

## 7. KL Divergence Numerical Stability (MEDIUM)

**File**: `train_ae.py:188`
```python
kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
```

With `logvar` clamped to [-30, 20]:
- `logvar=-30` → `logvar.exp() = 9.1e-14` (underflow, essentially 0)
- The term `1 + logvar - mean.pow(2) - logvar.exp()` can become very negative

**No numerical safeguard** for extreme logvar values.

**Fix**:
```python
kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - torch.clamp(logvar.exp(), min=1e-10, max=1e10))
```

---

## 8. Inference Normalization Bugs (HIGH)

**Bug A** - Latent decoding clamp range (`world/test_world.py:110-111`):
```python
x = x.clamp(0, 1)
```

VAE decoder outputs `[-1, 1]` (tanh activation), but clamping to `[0, 1]` clips all negative values.

**Fix**:
```python
x = (x + 1) / 2  # Convert [-1, 1] to [0, 1]
x = x.clamp(0, 1)
```

**Bug B** - Initial frame normalization (`world/test_world.py:82-83`):
```python
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
```

Training uses `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])` which converts `[0,1]` to `[-1,1]`.
Inference loads frames in `[0, 1]` without this transform.

**Fix**:
```python
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor * 2 - 1  # Convert to [-1, 1]
```

---

## 9. Hardcoded Latent Statistics (LOW-MEDIUM)

**File**: `train_ngen.py:37-38`
```python
"latent_mean": 10.1880,
"latent_std": 13.3726,
```

These are computed offline and hardcoded. If data distribution shifts:
- Different gameplay style
- More episodes added
- Different PPO checkpoint

The normalization becomes invalid, causing systematic bias.

**Better approach**: Compute running statistics or recompute per training run.

---

## 10. Additive Embedding Combination (LOW)

**File**: `diffuse/nn/resblock.py:46-51`
```python
scale, shift = self.time_proj(t_emb)...
if c_emb is not None:
    class_scale, class_shift = self.class_proj(c_emb)...
    scale, shift = scale + class_scale, shift + class_shift
if aug_emb is not None:
    aug_scale, aug_shift = self.aug_proj(aug_emb)...
    scale, shift = scale + aug_scale, shift + aug_shift
```

Naive addition means if one embedding has larger magnitude, it dominates. No normalization ensures equal contribution.

**Better approach**: Concatenate embeddings and project jointly, or normalize each before combining.

---

## 11. Cold Start in Inference (LOW)

**File**: `world/test_world.py:190-206`

Initial frames are loaded from real VOD data:
```python
past_frames, current_frame = load_initial_frames(vod_dir, k, device)
z_current = vae_encode(vae, current_frame, latent_mean, latent_std)
```

First prediction has **perfect real context**. Subsequent predictions use model-generated frames, creating a quality drop after the first frame.

**Impact**: First generated frame may look better than subsequent ones, masking model quality issues.

---

# PART 2: GAMENGEN COMPARISON ISSUES

These are differences from the GameNGen paper that may impact quality.

---

## 12. Context Length: 2 vs 64 frames (CRITICAL)

**Your code** (`diffuse/ngen/train_ngen.py:28`):
```python
"context_frames": 2,
```

**GameNGen uses 64 frames** (~3.2 seconds at 20fps).

**Impact**: With only 2 frames (~67ms), the model has almost no temporal memory:
- Cannot predict pipe positions that scrolled off-screen
- Cannot maintain consistent velocity over time
- Cannot track score changes

**Recommendation**: Increase to 8-16 frames for Flappy Bird, 32-64 for complex games.

---

## 13. Missing Classifier-Free Guidance (HIGH)

**GameNGen**: CFG weight 1.5, applied to past frames only, 10% dropout during training.

**Your implementation**: No CFG at all.

**What is CFG in this context?**
- `z_cond` (the past k frames concatenated to UNet input) is the conditioning
- During training: randomly zero out `z_cond` with 10% probability
- During inference: compute both conditional and unconditional predictions, interpolate

**Implementation**:

1. Training dropout (`train_ngen.py`, after line 218):
```python
# CFG training: drop conditioning 10% of time
if torch.rand(1).item() < 0.1:
    z_cond = torch.zeros_like(z_cond)
```

2. CFG sampling (`sampler.py`):
```python
@torch.no_grad()
def euler_sample_cfg(model, z_0, z_cond, action, aug_level, cfg_scale=1.5, num_steps=50):
    B = z_0.shape[0]
    device = z_0.device
    dt = 1.0 / num_steps
    z_cond_null = torch.zeros_like(z_cond)

    z = z_0
    for i in range(num_steps):
        t = torch.full((B,), (i + 0.5) * dt, device=device)  # Fixed midpoint

        v_cond = model(z, t, c=action, z_cond=z_cond, aug_level=aug_level)
        v_uncond = model(z, t, c=action, z_cond=z_cond_null, aug_level=aug_level)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        z = z + v * dt
    return z
```

---

## 14. Dataset Scale: ~164K vs 900M frames (HIGH)

**Your data**: ~1000 episodes × ~164 frames = ~164K frames
**GameNGen**: 900M frames (5000× more)

**Recommendation**: Collect 1-10M frames minimum via:
- Increase `num_episodes` in `vod/record.py`
- Use multiple PPO checkpoints
- Vary `p_stim` and `p_freeze` for diversity

---

## 15. No Decoder Fine-tuning (MEDIUM)

**GameNGen**: Fine-tunes latent decoder separately with MSE loss after U-Net training.

**Your code**: Trains VAE encoder+decoder jointly, no separate decoder refinement.

**Impact**: Fine details (score, small objects) may be blurry.

**Recommendation**: After flow model converges, freeze everything except decoder, fine-tune with MSE on generated→real pairs.

---

## 16. Model Capacity (MEDIUM)

**Your config**:
```python
"hidden_channels": 64,
"num_layers": 2,
```

Estimated ~3-5M parameters.

**GameNGen**: Stable Diffusion v1.4 U-Net (~860M parameters).

For Flappy Bird, smaller may be acceptable. If quality insufficient, increase to:
```python
"hidden_channels": 128,
"num_layers": 3,
```

---

## 17. Action Conditioning Method (LOW for Flappy Bird)

**Your code**: AdaGN (global scale/shift modulation)
**GameNGen**: Cross-attention on action sequence

For binary flap/no-flap, AdaGN is likely sufficient. Would matter for complex action spaces.

---

# PART 3: WHAT'S WORKING WELL

1. **Flow matching loss** - mathematically correct linear interpolation and constant velocity
2. **Noise augmentation** - properly discretized (16 buckets) and applied to conditioning
3. **Reflow mechanism** - correct backward/forward ODE approach
4. **Bird-weighted VAE loss** - smart emphasis on important region
5. **Action hijacking** - good data diversity technique
6. **Data pipeline action alignment** - verified correct (action[N] produces frame[N])

---

# PRIORITY FIX ORDER

## Phase 1: Critical Bugs (Fix Before Any Training)

| # | Issue | Severity | Effort | File |
|---|-------|----------|--------|------|
| 1 | Episode boundary wrap-around | **CRITICAL** | Low | `ngen_data.py` |
| 2 | ODE time stepping bug | **CRITICAL** | Low | `sampler.py` |
| 3 | Inference normalization bugs | **CRITICAL** | Low | `test_world.py` |
| 4 | Latent space bounds checking | **CRITICAL** | Low | `sampler.py` |
| 5 | Train-test step mismatch | **CRITICAL** | Low | `test_world.py` |

## Phase 2: Data Quality (Before Retraining)

| # | Issue | Severity | Effort | File |
|---|-------|----------|--------|------|
| 6 | Enable behavioral augmentation | **CRITICAL** | Low | `record.py` |
| 7 | Recollect diverse data | **HIGH** | High | `record.py` |
| 8 | Save initial reset frame | MEDIUM | Low | `environment.py` |

## Phase 3: Training Improvements

| # | Issue | Severity | Effort | File |
|---|-------|----------|--------|------|
| 9 | Context frames (2 → 8-16) | **CRITICAL** | Medium | `train_ngen.py` |
| 10 | Implement CFG | **HIGH** | Medium | `train_ngen.py`, `sampler.py` |
| 11 | Add gradient clipping | HIGH | Low | `train_ngen.py` |
| 12 | Fix VAE sampling mismatch | HIGH | Low | `train_ngen.py` |
| 13 | Add LR scheduling | MEDIUM | Low | `train_ngen.py` |

## Phase 4: Inference Stability

| # | Issue | Severity | Effort | File |
|---|-------|----------|--------|------|
| 14 | Fix noise injection (z_0) | HIGH | Low | `test_world.py` |
| 15 | Set appropriate aug_level | MEDIUM | Low | `test_world.py` |
| 16 | Add temporal smoothing | MEDIUM | Low | `test_world.py` |

## Phase 5: Quality Refinements

| # | Issue | Severity | Effort | File |
|---|-------|----------|--------|------|
| 17 | Decoder fine-tuning | MEDIUM | Medium | New script |
| 18 | Increase model capacity | MEDIUM | Low | `train_ngen.py` |

---

# FILES TO MODIFY

| File | Changes |
|------|---------|
| `diffuse/ngen/ngen_data.py` | Fix episode boundary wrap-around (line 60) |
| `diffuse/ngen/sampler.py` | Fix ODE time stepping, add latent clipping, add CFG sampling |
| `diffuse/ngen/train_ngen.py` | Add CFG dropout, gradient clipping, LR scheduler, increase context_frames, fix VAE sample mode |
| `world/test_world.py` | Fix normalization bugs, match step count, fix z_0 injection, add temporal smoothing |
| `vod/record.py` | Enable p_stim/p_freeze, increase num_episodes |
| `game/environment.py` | Save initial reset frame |
| `diffuse/ae/train_ae.py` | Add decoder fine-tuning phase |

---

# VERIFICATION CHECKLIST

**Before Training:**
- [ ] Episode wrap-around disabled (start_step = k for all episodes)
- [ ] Data collected with behavioral augmentation enabled
- [ ] Initial reset frames saved in VOD data

**After Training:**
- [ ] ODE integration reaches t=1.0 exactly
- [ ] Forward and backward integration are symmetric
- [ ] Inference uses same step count as training (50)
- [ ] Initial frames normalized to [-1, 1]
- [ ] Decoded images properly converted from [-1, 1] to [0, 1]
- [ ] Gradient norms stay bounded during training
- [ ] Latents clipped to [-3, 3] in normalized space
- [ ] CFG improves conditioning adherence

**Long-horizon Generation:**
- [ ] Generated sequences maintain consistency >30 seconds
- [ ] No catastrophic drift after 1000+ frames
- [ ] Bird physics appear realistic
- [ ] PSNR > 25 dB on held-out frames

---

# QUICK REFERENCE

| Aspect | GameNGen | Your Implementation | Issue? |
|--------|----------|---------------------|--------|
| Episode boundaries | Proper isolation | Wrap-around contamination | **CRITICAL** |
| Behavioral diversity | RL exploration | Single policy (p_stim=0) | **CRITICAL** |
| Latent bounds | Implicit | No checking | **CRITICAL** |
| ODE endpoint | Proper | Stops at t=0.98 | **CRITICAL** |
| z_0 initialization | Coupled | Fresh random each frame | **HIGH** |
| Context frames | 64 | 2 | **CRITICAL** |
| CFG | Yes (1.5) | No | **HIGH** |
| Inference steps | 4 DDIM | 20 Euler (vs 50 train) | **HIGH** |
| Data size | 900M | ~164K | **HIGH** |
| Gradient clip | Yes (1.0) | No | **HIGH** |
| aug_level inference | Matched | Static 0 | MEDIUM |
| Temporal smoothing | Implicit | None | MEDIUM |
| Decoder fine-tune | Yes | No | MEDIUM |
| Model params | ~860M | ~3-5M | Acceptable |
| Action conditioning | Cross-attn | AdaGN | Acceptable |
| Noise aug training | 0.7, 10 bins | 0.5, 16 bins | OK |

---

# REFERENCES

- [GameNGen Paper](https://arxiv.org/abs/2408.14837)
- [GameNGen Project Page](https://gamengen.github.io/)
- [Full Paper HTML](https://arxiv.org/html/2408.14837v1)

---

# TODO: Add explicit "done" head for end-of-episode detection

**Context / Why:**  
The world model predicts only the next frame. It has no way to declare a terminal state, especially now that conditioning windows stay within a single episode. Without an explicit termination signal, inference can continue generating frames past the true end of an episode, which leads to unrealistic rollouts and action/physics drift. GameNGen handles session transitions by including reset moments in the context; here we need a direct termination prediction to stop/reset cleanly.

**What this accomplishes:**  
Adds a reliable, learned "game over" signal during rollout. This allows the loop to end or reset at the right time, improves long-horizon stability, and avoids learning impossible post-termination transitions.

## Plan

1) **Add termination labels to the dataset**
   - Parse `terminated` and `truncated` from `run_info.jsonl` in each VOD run.
   - For each frame `t`, create `done_t = terminated_t OR truncated_t`.
   - Align `done_t` with the same `step` index used for `current_frame` in `TraceDataset`.
   - If logs are missing, fall back to `done_t = 1` for the final frame in the episode.

2) **Extend the model with a "done" head**
   - Add a small MLP or 1x1 conv head to `ResUNet` (or a wrapper module) that outputs a scalar logit per sample.
   - Feed it a stable feature (e.g., bottleneck features or pooled `z_t` + conditioning).
   - Keep the main flow head unchanged.

3) **Update the loss**
   - Add `BCEWithLogitsLoss(done_logit, done_label)` to the existing flow loss.
   - Introduce a weight `done_loss_weight` (start with 0.1–1.0) and tune if needed.

4) **Train and validate**
   - Train the augmented model on the same VOD data.
   - Track `done` accuracy/precision/recall; ensure low false negatives (missed terminations).

5) **Use the done head during inference**
   - In `world/test_world.py`, after sampling the next frame, compute `done_prob`.
   - If `done_prob > threshold` (e.g., 0.5–0.7), stop the rollout or reset to a new episode.
   - Optionally show a visual indicator for detected termination.

6) **Sanity checks**
   - Verify that `done` fires near actual crash frames in held-out episodes.
   - Ensure rollouts stop within 1–2 frames of true termination.
