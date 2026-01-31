# Flappy Bird World Model - Implementation Review vs GameNGen

## IMPORTANT INSTRUCTIONS - KEEP IN CONTEXT

For now we are ONLY working on the diffuse/ directory and the models inside it. We can propose changes to game/ and vod/ (data collection) later. 

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

## 7. KL Divergence Numerical Stability (MEDIUM)

**File**: `train_vae.py:188`
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

**Recommendation**: Increase to 8 frames. 

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


## 15. No Decoder Fine-tuning (MEDIUM)

**GameNGen**: Fine-tunes latent decoder separately with MSE loss after U-Net training.

**Your code**: Trains VAE encoder+decoder jointly, no separate decoder refinement.

**Impact**: Fine details (score, small objects) may be blurry.

**Recommendation**: After flow model converges, freeze everything except decoder, fine-tune with MSE on generated→real pairs.

**DONT WORRY ABOUT THIS YET**

---

## 16. Model Capacity (LOW)

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

**DONT WORRY ABOUT THIS RIGHT NOW. DONT  CHANGE MY MODEL SIZE**

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

## NOW: diffuse/ and world/ Changes

### Phase 1: Critical Bugs (Fix Immediately)

| # | Issue | Severity | Effort | File | Status |
|---|-------|----------|--------|------|--------|
| 1 | Episode boundary wrap-around | **CRITICAL** | Low | `diffuse/ngen/ngen_data.py` | ✅ Already fixed |
| 2 | ODE time stepping bug | **CRITICAL** | Low | `diffuse/ngen/sampler.py` | ✅ DONE |
| 3 | Inference normalization bugs | **CRITICAL** | Low | `world/test_world.py` | ✅ Already fixed |
| 4 | Latent space bounds checking | **CRITICAL** | Low | `diffuse/ngen/sampler.py` | ✅ DONE |
| 5 | Train-test step mismatch | **CRITICAL** | Low | `world/test_world.py` | ✅ DONE |

### Phase 2: Training Improvements

| # | Issue | Severity | Effort | File | Status |
|---|-------|----------|--------|------|--------|
| 6 | Context frames (2 → 8-16) | **CRITICAL** | Medium | `diffuse/ngen/train_ngen.py` | ✅ DONE (8 frames) |
| 7 | Implement CFG | **HIGH** | Medium | `diffuse/ngen/train_ngen.py`, `diffuse/ngen/sampler.py` | ✅ DONE |
| 8 | Add gradient clipping | HIGH | Low | `diffuse/ngen/train_ngen.py` | ✅ DONE |
| 9 | Fix VAE sampling mismatch | HIGH | Low | `diffuse/ngen/train_ngen.py` | N/A |
| 10 | Add LR scheduling | MEDIUM | Low | `diffuse/ngen/train_ngen.py` | ✅ DONE |

### Phase 3: Inference Stability

| # | Issue | Severity | Effort | File | Status |
|---|-------|----------|--------|------|--------|
| 11 | Fix noise injection (z_0) | HIGH | Low | `world/test_world.py` | ✅ DONE |
| 12 | Set appropriate aug_level | MEDIUM | Low | `world/test_world.py` | ✅ DONE |
| 13 | Add temporal smoothing | MEDIUM | Low | `world/test_world.py` | ✅ DONE |

### Phase 4: Quality Refinements

| # | Issue | Severity | Effort | File | Status |
|---|-------|----------|--------|------|--------|
| 14 | Decoder fine-tuning | MEDIUM | Medium | `diffuse/vae/train_vae.py` (new phase) | ⏳ Deferred |
| 15 | Increase model capacity | MEDIUM | Low | `diffuse/ngen/train_ngen.py` | ⏳ Deferred |

---

## LATER: Data Collection Changes (vod/, game/)

These changes should be done separately before retraining:

| # | Issue | Severity | File |
|---|-------|----------|------|
| L1 | Enable behavioral augmentation (p_stim, p_freeze) | **CRITICAL** | `vod/record.py` |
| L2 | Recollect diverse data (10K+ episodes) | **HIGH** | `vod/record.py` |
| L3 | Save initial reset frame | MEDIUM | `game/environment.py` |

---

# FILES MODIFIED (Current Focus: diffuse/ and world/)

| File | Changes | Status |
|------|---------|--------|
| `diffuse/ngen/ngen_data.py` | Fix episode boundary wrap-around, add done label parsing | ✅ Done |
| `diffuse/ngen/sampler.py` | Fix ODE time stepping, add latent clipping, add CFG sampling | ✅ Done |
| `diffuse/ngen/train_ngen.py` | Add CFG dropout, gradient clipping, LR scheduler, context_frames=8, done head training | ✅ Done |
| `diffuse/ngen/loss.py` | Add done loss term (BCE) | ✅ Done |
| `diffuse/nn/resunet.py` | Add done head MLP for termination detection | ✅ Done |
| `diffuse/vae/train_vae.py` | Add decoder fine-tuning phase | ⏳ Deferred |
| `world/test_world.py` | Fix normalization, match step count, fix z_0 injection, add smoothing, add done detection, add CFG | ✅ Done |

---

# VERIFICATION CHECKLIST

**Before Training:**
- [x] Episode wrap-around disabled (start_step = k for all episodes) ✅
- [ ] Data collected with behavioral augmentation enabled (deferred)
- [ ] Initial reset frames saved in VOD data (deferred)

**Code Implementation (complete):**
- [x] ODE integration uses midpoint rule (t = (i+0.5)*dt) ✅
- [x] Forward and backward integration are symmetric ✅
- [x] Inference uses same step count as training (50) ✅
- [x] Initial frames normalized to [-1, 1] ✅
- [x] Decoded images properly converted from [-1, 1] to [0, 1] ✅
- [x] Gradient clipping enabled (max_norm=1.0) ✅
- [x] Latents clipped to [-3, 3] in normalized space ✅
- [x] CFG implemented (10% dropout training, euler_sample_cfg) ✅
- [x] LR scheduling (cosine annealing) ✅
- [x] Context frames = 8 ✅
- [x] Done head implemented ✅

**After Retraining (to verify):**
- [ ] Gradient norms stay bounded during training
- [ ] CFG improves conditioning adherence
- [ ] Done head fires near crash frames

**Long-horizon Generation (to verify):**
- [ ] Generated sequences maintain consistency >30 seconds
- [ ] No catastrophic drift after 1000+ frames
- [ ] Bird physics appear realistic
- [ ] PSNR > 25 dB on held-out frames

---

# QUICK REFERENCE

| Aspect | GameNGen | Your Implementation | Status |
|--------|----------|---------------------|--------|
| Episode boundaries | Proper isolation | Wrap-around contamination | ✅ Fixed |
| Behavioral diversity | RL exploration | Single policy (p_stim=0) | ⏳ Deferred |
| Latent bounds | Implicit | No checking | ✅ Fixed |
| ODE endpoint | Proper | Stops at t=0.98 | ✅ Fixed (midpoint rule) |
| z_0 initialization | Coupled | Fresh random each frame | ✅ Fixed (perturbation) |
| Context frames | 64 | 2 | ✅ Fixed (8 frames) |
| CFG | Yes (1.5) | No | ✅ Fixed |
| Inference steps | 4 DDIM | 20 Euler (vs 50 train) | ✅ Fixed (50 steps) |
| Data size | 900M | ~164K | ⏳ Deferred |
| Gradient clip | Yes (1.0) | No | ✅ Fixed |
| aug_level inference | Matched | Static 0 | ✅ Fixed (default 8) |
| Temporal smoothing | Implicit | None | ✅ Fixed (optional) |
| Decoder fine-tune | Yes | No | ⏳ Deferred |
| Model params | ~860M | ~3-5M | Acceptable |
| Action conditioning | Cross-attn | AdaGN | Acceptable |
| Noise aug training | 0.7, 10 bins | 0.5, 16 bins | OK |
| Done head | N/A | N/A | ✅ Implemented |
| LR scheduling | Yes | No | ✅ Fixed (cosine) |

---

# REFERENCES

- [GameNGen Paper](https://arxiv.org/abs/2408.14837)
- [GameNGen Project Page](https://gamengen.github.io/)
- [Full Paper HTML](https://arxiv.org/html/2408.14837v1)

---

# ✅ DONE: Add explicit "done" head for end-of-episode detection

**Status: IMPLEMENTED**

**What was implemented:**

1) ✅ **Add termination labels to the dataset** (`diffuse/ngen/ngen_data.py`)
   - Parse `terminated` and `truncated` from `run_info.jsonl` in each VOD run.
   - For each frame `t`, create `done_t = terminated_t OR truncated_t`.
   - Falls back to `done_t = 1` for the final frame in the episode.
   - Dataset returns done labels when `include_done=True`.

2) ✅ **Extend the model with a "done" head** (`diffuse/nn/resunet.py`)
   - Added MLP head from bottleneck features: `AdaptiveAvgPool2d → Linear(64) → SiLU → Linear(1)`.
   - Enabled via `use_done_head=True` constructor arg.
   - `forward(..., return_done=True)` returns `(v_pred, done_logit)`.

3) ✅ **Update the loss** (`diffuse/ngen/loss.py`)
   - Added `BCEWithLogitsLoss(done_logit, done_label)` to flow loss.
   - Configurable `done_loss_weight` (default 0.1).

4) ✅ **Training integration** (`diffuse/ngen/train_ngen.py`)
   - Added config: `use_done_head: True`, `done_loss_weight: 0.1`.
   - Dataset loaded with `include_done=True`.
   - Training loop unpacks done labels and passes to loss function.
   - Logging includes `done_loss`.

5) ✅ **Inference integration** (`world/test_world.py`)
   - Added `--done-threshold` CLI arg (default 0.5).
   - After sampling, checks `done_prob` and prints "Game Over!" if above threshold.

6) **Sanity checks** - To be verified after retraining:
   - [ ] Verify that `done` fires near actual crash frames in held-out episodes.
   - [ ] Ensure rollouts stop within 1–2 frames of true termination.

---

# 🔴 CRITICAL: Action Class Imbalance (Model Ignores Actions)

**Status: NEEDS FIX BEFORE RETRAINING**

## The Problem

At 65 epochs, the world model performs poorly - bird doesn't follow physics, doesn't respond to player input. Investigation revealed:

### Action Distribution Analysis
```
Total steps: 164,104
Action 0 (no-flap): 155,173 (94.6%)
Action 1 (flap):      8,931 (5.4%)
```

The training data is **severely imbalanced** - 17x more no-flap than flap samples.

### Diagnostic: action_diff Test

Added debug logging in `test_world.py` that computes how different the model's prediction is when using opposite actions:

```python
# Run same frame with action=0 vs action=1
z_next_action0 = euler_sample(model, z_0, z_cond, action=0, ...)
z_next_action1 = euler_sample(model, z_0, z_cond, action=1, ...)
action_diff = torch.abs(z_next_action0 - z_next_action1).mean()
```

**Results:**
```
Frame 60:  action_diff=0.0131
Frame 70:  action_diff=0.0161
Frame 80:  action_diff=0.0096
Frame 90:  action_diff=0.0302
Frame 100: action_diff=0.0139
```

**action_diff ≈ 0.01-0.03** → Model produces nearly identical outputs regardless of action input!

For comparison, `frame_diff` (difference between consecutive predicted frames) was ~0.17, so action_diff should be at least 0.1+ if the model were conditioning properly.

### Root Cause

With 94.6% of training data being action=0, the model learns to simply predict "what happens with no action" and essentially ignores the action conditioning input. The action embedding becomes a dead weight.

## Proposed Fix: Loss Reweighting

Instead of oversampling (which inflates dataset size and repeats data), use **loss reweighting** to make action=1 samples contribute more to the gradient:

```python
# In train_ngen.py, compute sample weights based on action
action_weights = torch.where(
    actions == 1,
    torch.tensor(17.0, device=device),  # Upweight flap (minority)
    torch.tensor(1.0, device=device)    # Normal weight for no-flap
)

# Apply per-sample weighting to flow loss
flow_loss_per_sample = ((v_pred - v_target) ** 2).mean(dim=[1, 2, 3])
weighted_flow_loss = (flow_loss_per_sample * action_weights).mean()
```

The weight of 17.0 corresponds to the imbalance ratio (155173 / 8931 ≈ 17.4).

Alternatively, use focal-style weighting or dynamically compute weights per batch.

## Other Fixes Applied

### Frame Buffer Timing Bug (Fixed)

The inference loop in `test_world.py` was updating the frame buffer AFTER sampling instead of BEFORE:

```python
# BUGGY (was):
z_next = euler_sample(model, z_0, z_cond, ...)  # z_cond missing current frame!
frame_buffer.pop(0)
frame_buffer.append(z_display.clone())          # Added AFTER sampling

# FIXED (now):
frame_buffer.pop(0)
frame_buffer.append(z_display.clone())          # Update FIRST
z_cond = torch.cat(frame_buffer, dim=1)         # Now includes current frame
z_next = euler_sample(model, z_0, z_cond, ...)  # Correct context
```

Same fix applied to `web/src/App.ts`.

### Default aug_level Changed (Fixed)

Changed default `aug_level` from 8 to 0 for cleaner inference (less noise on conditioning).

## Verification After Fix

After retraining with loss reweighting:
1. **action_diff should increase** to 0.1+ (model predictions differ based on action)
2. **Bird should respond to input** - jump when SPACE pressed, fall when idle
3. **Smooth physics** - gravity pulls bird down, flap gives upward velocity

## Files to Modify

| File | Change |
|------|--------|
| `diffuse/ngen/train_ngen.py` | Add loss reweighting based on action |
| `diffuse/ngen/loss.py` | Support per-sample weighted flow loss |

---

# Optimizations 1/27

Memory and performance optimizations applied:

| File | Change | Impact |
|------|--------|--------|
| `diffuse/vae/train_vae.py` | Added bfloat16 autocast (was missing, ngen already had it) | ~2x memory, ~30% speedup |
| `diffuse/nn/resunet.py` | Added gradient checkpointing to forward pass | ~25% compute cost for ~50% memory reduction |
| `diffuse/nn/resblock.py` | Replaced ConvTranspose2d with interpolate+conv in UpResBlock | Lower memory, avoids checkerboard artifacts |

**Note:** The UpResBlock change affects both VAE decoder and ResUNet. Existing checkpoints won't load (different weight shapes) - requires retraining.

---

# Optimizations 1/27 (Part 2)

Additional performance optimizations:

| File | Change | Impact |
|------|--------|--------|
| `diffuse/vae/train_vae.py` | Added `prefetch_factor=2`, `drop_last=True`, `non_blocking=True`, `torch.compile(mode="reduce-overhead")` | Better GPU utilization, static shapes for compile |
| `diffuse/ngen/train_ngen.py` | Same DataLoader + memory format + compile optimizations | Same benefits |
| `diffuse/ngen/sampler.py` | Changed `@torch.no_grad()` to `@torch.inference_mode()` | Slightly faster inference |
| `latent-vod/encode_vod.py` | Changed `torch.no_grad()` to `torch.inference_mode()` | Slightly faster encoding |
| `vod/record.py` | Changed `torch.no_grad()` to `torch.inference_mode()` | Slightly faster inference |

**Note:** `torch.compile` with `reduce-overhead` uses CUDA graphs. May still be memory-expensive - if so, try `mode="default"` instead.

---

# Removed channels_last Memory Format (1/27)

Removed all `memory_format=torch.channels_last` optimizations from the codebase:

- **Reason**: Caused `RuntimeError: required rank 4 tensor to use channels_last format` when applied to rank 5 tensors (`past_frames` with shape `[batch, k, channels, H, W]`). This is a late-stage optimization that's not essential for correctness.

- **Files modified**:
  - `diffuse/ngen/train_ngen.py`: Removed from model `.to(device)` and tensor transfers (3 occurrences)
  - `diffuse/vae/train_vae.py`: Removed from model `.to(device)` and batch transfers (2 occurrences)

- **Impact**: Minimal performance difference expected. The default NCHW memory format works fine for this use case.

---

# Automatic Latent Statistics Computation (1/27)

Automated the process of computing and saving latent mean/std statistics after VAE training:

- **What changed**:
  - `diffuse/vae/train_vae.py`: Added `compute_latent_statistics()` function that samples 500 random frames and computes latent mean/std. Automatically called after training completes and saves results to `config.json` in the run directory.
  - `latent-vod/encode_vod.py`: Added `load_latent_statistics()` function that automatically loads statistics from the run directory's `config.json` based on checkpoint path. Falls back to default values if config not found (backward compatible).

- **Benefits**:
  - No more manual running of `compute_latent_std.py` and copying values
  - Statistics automatically stored alongside training config for easy tracking
  - Encoding script automatically uses correct statistics for each VAE checkpoint

- **Files modified**:
  - `diffuse/vae/train_vae.py`: Added statistics computation function and post-training hook
  - `latent-vod/encode_vod.py`: Added statistics loading and updated encode functions to use loaded values

---

# Automatic Distribution Weight Computation (1/27)

Automated the computation of `action_weight` and `done_pos_weight` from actual dataset statistics:

- **What changed**:
  - `latent-vod/encode_vod.py`: Added `compute_dataset_statistics()` function that samples ~100 runs per category directory, counts action=0/1 and done=0/1 occurrences, and computes weights as inverse ratios. Called at end of encoding and saves results to `encode_config.json`.
  - `diffuse/ngen/train_ngen.py`: Added `load_weights_from_encode_config()` function that reads `action_weight` and `done_pos_weight` from `encode_config.json` when `--latent-vod` is provided, overriding the hardcoded defaults.

- **Benefits**:
  - No more manual computation and hardcoding of distribution weights
  - Weights automatically adapt if dataset composition changes
  - Statistics stored in `encode_config.json` for transparency and reproducibility

- **Expected `encode_config.json` output**:
  ```json
  {
    "vae_checkpoint": "/path/to/vae.pt",
    "latent_mean": 0.4755,
    "latent_std": 1.5959,
    "action_weight": 17.22,
    "done_pos_weight": 159.10,
    "statistics": {
      "total_steps": 160104,
      "action_0_count": 151316,
      "action_1_count": 8788,
      "done_0_count": 159104,
      "done_1_count": 1000
    }
  }
  ```

- **Files modified**:
  - `latent-vod/encode_vod.py`: Added `compute_dataset_statistics()` and post-encoding hook
  - `diffuse/ngen/train_ngen.py`: Added `load_weights_from_encode_config()` and auto-loading logic

---

# 📋 2026-01-30: World Model Quality Improvement Plan

## Current Symptoms
- Images are sharp (VAE is fine)
- Bird **ignores gravity** — floats in place rather than falling when no flap
- Bird **occasionally disappears** entirely

## Root Cause Analysis

### Primary cause: CFG dropout zeros out actions (train_ngen.py:248-251)

The current CFG dropout replaces **both** `z_cond` (frame conditioning) AND `actions` with null values. GameNGen explicitly only drops frame conditioning, never actions (Section 3.2 of the paper).

**Why this breaks things:**
- The unconditional prediction `v_uncond` has NO action information
- At inference, CFG formula: `v = v_uncond + 1.5 * (v_cond - v_uncond)`
- This conflates "unconditional on frames" with "unconditional on actions"
- The model's action-following signal gets diluted by CFG amplification
- When `v_uncond` predicts "no bird" (because it has zero context), CFG combination can erase the bird

### Secondary causes
- **Model too small** — 3M params, 2 layers (limited by odd latent width 36 → 9 after 2 downsamples)
- **Short context** — 8 frames (0.27s) vs GameNGen's 64 frames (3.2s)
- **Dataset size** — 141K frames vs GameNGen's 70M
- **Noise augmentation** — max_std=0.5 vs GameNGen's 0.7

## Plan: Three Phases

### Phase 1: Fix CFG Dropout (HIGHEST PRIORITY — isolate primary cause)

**Rationale**: 2-line code fix that isolates the most likely root cause. Retrain with everything else unchanged.

**File: `diffuse/ngen/train_ngen.py` lines 248-251**
```python
# CURRENT (broken):
cfg_mask = torch.rand(B, device=device) < train_config["cfg_dropout_prob"]
z_cond = torch.where(cfg_mask.view(B, 1, 1, 1), torch.zeros_like(z_cond), z_cond)
actions = torch.where(cfg_mask.view(B, 1), torch.full_like(actions, model_config["num_classes"]), actions)

# FIX: Remove the action nullification line. Only zero out z_cond:
cfg_mask = torch.rand(B, device=device) < train_config["cfg_dropout_prob"]
z_cond = torch.where(cfg_mask.view(B, 1, 1, 1), torch.zeros_like(z_cond), z_cond)
# DO NOT null out actions — CFG should only drop frame conditioning
```

**File: `diffuse/ngen/sampler.py` — euler_sample and euler_sample_backward**
```python
# CURRENT (broken):
actions_null = torch.full_like(actions, num_classes)
v_uncond, _ = model(z, t, z_cond=z_cond_null, c=actions_null, aug_level=aug_level)

# FIX: Pass real actions to unconditional path:
v_uncond, _ = model(z, t, z_cond=z_cond_null, c=actions, aug_level=aug_level)
```

**Verification**: Retrain → run `test_world.py --debug`. Bird should follow gravity and not disappear.

---

### Phase 2: Increase Model Capacity (if Phase 1 alone isn't sufficient)

**2a. Pad latents to fix odd-dimension bottleneck**

File: `diffuse/nn/resunet.py`
- Reflect-pad latent width from 36 → 40 at start of `forward()`
- Crop output back to 36 after `out_conv`
- This enables 3 downsample layers: 64×40 → 32×20 → 16×10 (all even)

**2b. Scale up model**

File: `diffuse/ngen/train_ngen.py`
- `num_layers`: 2 → 3
- `hidden_channels`: 64 → 128
- Estimated ~25-30M params (still fine for H200)

**2c. Add self-attention at bottleneck**

File: `diffuse/nn/resunet.py`
- Add `nn.MultiheadAttention` after `self.bot`
- At 16×10 spatial (160 tokens), self-attention is cheap
- Enables global reasoning about bird position relative to pipes

---

### Phase 3: Training Improvements (apply together with Phase 2)

**3a. Increase noise augmentation** (train_ngen.py)
- `max_aug_std`: 0.5 → 0.7
- `num_aug_bins`: 16 → 10 (match GameNGen)

**3b. Increase context window** (train_ngen.py)
- `context_size`: 8 → 16 (doubles temporal context to 0.53s)

**3c. Collect more training data** (vod/record.py)
- Record 5-10× more VOD episodes (target ~700K-1M+ frames)
- Add higher `p_stim` values (0.02, 0.03) for more flap-heavy episodes
- Re-encode all latents

---

## Execution Order

1. **Phase 1 only** → retrain → test. Isolates whether CFG is the primary cause.
2. If Phase 1 alone isn't enough: **Phase 1 + 2 + 3** → retrain → test.

## Files to Modify

| File | Phase | Change |
|------|-------|--------|
| `diffuse/ngen/train_ngen.py` | 1 | Remove action nullification from CFG dropout |
| `diffuse/ngen/sampler.py` | 1 | Pass real actions in unconditional CFG path |
| `diffuse/nn/resunet.py` | 2 | Latent padding, 3 layers, self-attention |
| `diffuse/ngen/train_ngen.py` | 2+3 | Model config + aug + context changes |
| `vod/record.py` | 3 | More episodes, higher p_stim values |

## Key Reference: GameNGen vs Current Implementation

| Aspect | GameNGen | Current | Proposed |
|--------|----------|---------|----------|
| CFG drops actions? | **No** (frames only) | Yes (both) | **No** (Phase 1) |
| Model params | ~860M | 3M | ~25-30M (Phase 2) |
| Downsample layers | Many | 2 | 3 (Phase 2) |
| Self-attention | Yes | No | Yes at bottleneck (Phase 2) |
| Noise aug max | 0.7 | 0.5 | 0.7 (Phase 3) |
| Context frames | 64 | 8 | 16 (Phase 3) |
| Training data | 70M | 141K | ~700K-1M (Phase 3) |

---

# Death Frames / Terminated Flag Mismatch (1/31)

## Current Behavior

| Frame | `terminated` | Overlay | Visual content |
|-------|-------------|---------|----------------|
| Collision frame | `True` | No | Normal game (bird hitting pipe) |
| Death frames 1-5 | `True` | Yes | Game-over overlay |
| **Total** | **6 with True** | **5 with overlay** | |

- `DEATH_FRAMES = 5` in `game/environment.py`
- The collision frame is labeled `terminated=True` but looks like normal gameplay (no overlay)
- The 5 death frames are labeled `terminated=True` and have the gameover overlay sprite

## Problems

1. **Mixed visual signal for done head**: The collision frame looks like normal gameplay but is labeled terminated. The 5 overlay frames look visually distinct. The done head gets confused about what "terminated" looks like.

2. **Wasted generation capacity**: Once the done head fires during inference (`test_world.py`), generation stops immediately. The model never needs to *generate* overlay frames. Every death-frame sample teaches the generation head something it'll never use.

3. **Context pollution**: With k=8, a sample at death frame 5 has context that's ~62% death frames (5/8). With k=16 it's ~31%. These are distributions the model never encounters during inference since it stops at the first `done=True`.

4. **Class weight distortion**: `done_pos_weight` is computed as `alive_count / dead_count`. Having 6 terminated frames per episode instead of 1-2 makes the imbalance look less severe, which under-weights the done signal during training.

## Recommendation

Reduce death frames to 1-2 total:

- **1 terminated frame** (the collision frame, no overlay) is the cleanest signal — "this is what it looks like when the game ends."
- If keeping overlay frames at all, 1-2 is sufficient without polluting context windows or wasting generation capacity.
- Keep `terminated=True` consistent with visual content — either all overlay frames are terminated, or the collision frame without overlay is the only terminated one. The current split (1 without overlay + 5 with) is the worst of both worlds for the done head.

## Action Items

- [ ] Reduce `DEATH_FRAMES` from 5 to 1-2 in `game/environment.py`
- [ ] Decide: overlay on terminated frames, or no overlay at all (cleanest for model)
- [ ] Re-collect data after change
- [ ] Recompute `done_pos_weight` (will increase since fewer done=1 frames)

---
