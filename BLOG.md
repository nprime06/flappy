# Building a Neural Game Engine for Flappy Bird

This is a writeup of an ongoing project to build a playable neural game engine for Flappy Bird — a model that replaces the game's logic and rendering entirely with a neural network. Given a history of frames and the player's actions, the model predicts what the next frame should look like, pixel by pixel. The player interacts with the model in real time through a keyboard, and the model autoregressively generates the game.

The project is inspired by GameNGen (Valevski et al., 2024), which demonstrated this idea on DOOM using a fine-tuned Stable Diffusion model. GameNGen showed that a diffusion model, conditioned on 64 past frames and player actions, could simulate DOOM at 20 FPS with visual quality that human raters could barely distinguish from the real game. Their key ingredients were a pretrained Stable Diffusion 1.4 U-Net (~860M parameters), 900 million frames of gameplay collected by an RL agent, noise augmentation on conditioning frames to stabilize autoregressive generation, and classifier-free guidance applied only to frame conditioning.

This project takes the same core idea but makes several deliberate departures. Instead of fine-tuning a large pretrained model, we train everything from scratch — a custom VAE, a custom U-Net, and the flow model itself — on a fraction of the data (~470K frames). Instead of denoising diffusion with DDIM sampling, we use flow matching with Euler integration. Instead of DOOM's complex 3D environment, we use Flappy Bird: a simpler game, but one that still requires the model to learn gravity, collision, scrolling, and action-conditioned dynamics. The goal is to understand what it actually takes to make a neural game engine work, in a setting where every component is transparent and every design choice is ours.

```
[DIAGRAM: Side-by-side comparison of a real Flappy Bird game frame and the world model's
generated output for the same game state. Include several frames showing bird at different
heights, pipes scrolling, and a game-over frame if available.]
```

## System Overview

The system is a five-stage pipeline. First, a PPO agent learns to play Flappy Bird, producing a small policy network. Second, we record diverse gameplay by running this agent with randomized action hijacking — occasionally forcing flaps or suppressing them to explore states the optimal policy would never visit. Third, a VAE compresses raw game frames into a compact latent representation. Fourth, we encode all recorded frames into this latent space. Finally, a flow matching model learns to predict the next latent frame given a window of past latent frames and the corresponding player actions.

At inference time, the flow model runs autoregressively: it takes its own previous outputs as conditioning, accepts real-time keyboard input as the action, generates the next latent, decodes it through the VAE, and displays it on screen. The player sees and interacts with a game that exists entirely inside the neural network.

```
[DIAGRAM: Pipeline flow diagram showing the five stages:
PPO Training -> VOD Recording (with hijacked agent) -> VAE Training -> Latent Encoding -> Flow Model Training
With a separate inference path: Keyboard Input + Frame Buffer -> Flow Model -> VAE Decoder -> Display]
```

### Data Collection and Behavioral Diversity

A naive approach would record an expert agent playing optimally and train on those trajectories. But an optimal Flappy Bird agent visits a narrow slice of the state space: the bird hovers near pipe gaps, rarely ventures to extreme heights, and almost never crashes in interesting ways. A world model trained on this data would have no idea what gravity looks like when the bird is far from a gap, or what happens when the player mashes the flap button.

To address this, we use what we call a "hijacked" agent. The trained PPO policy proposes actions normally, but with probability `p_stim` a no-flap is overridden to a flap, and with probability `p_freeze` a flap is suppressed to a no-flap. By recording many episodes across a grid of these probabilities — from the vanilla policy at (0, 0) to a heavily randomized policy at (0.05, 0.5) — we collect data that covers the full state space: birds flying too high, crashing into the ground, flapping erratically near pipes, and everything in between.

The latest data collection produced about 467,000 frames (4.3 hours of gameplay) across 3,135 episodes and 17 hijacking configurations. This is roughly 500x less data than GameNGen used for DOOM, which creates real pressure on every other design choice to be data-efficient.

## The Latent Space

Training a generative model directly on 288x512 RGB images would be computationally prohibitive, especially at the data scales we're working with. Following the latent diffusion paradigm, we first train a VAE to compress frames into a much smaller latent representation, then train the flow model entirely in this latent space.

### VAE Architecture and Compression

The VAE is a straightforward encoder-decoder with residual blocks and strided convolutions for downsampling. With 3 downsampling layers and 4 latent channels, a 288x512x3 image compresses to a 36x64x4 latent tensor — roughly a 120x reduction in dimensionality. The encoder has about 650K parameters and the decoder about 370K, making it small enough to train in a few hours on two H200 GPUs.

One important design choice: we normalize the latent space externally rather than baking it into the VAE. After training, we compute the empirical mean and standard deviation over a sample of encoded frames (approximately 0.48 and 1.59 respectively), and apply z-score normalization during the encoding step. This means the flow model always sees latents centered at zero with unit variance, regardless of which VAE checkpoint produced them. It's a small thing, but it decouples the two training stages cleanly — we can swap VAE checkpoints without retuning the flow model's loss landscape.

### Bird-Weighted Reconstruction

Flappy Bird frames are dominated by static background — sky, ground, and pipes that scroll predictably. The bird itself occupies a tiny fraction of the image, but it's the most important element for gameplay. A standard reconstruction loss would happily sacrifice bird fidelity to reduce error on the vast background regions.

To address this, the VAE training loss applies a 10x weight multiplier to a rectangular region around the bird. The bird is detected using a simple hardcoded color mask — it's orange, so we threshold on (R > 0.8, 0.3 < G < 0.6, B < 0.3) to find bird pixels, then expand the bounding box slightly to capture the white head and wing details. This is crude but effective: the decoder learns to reconstruct the bird sharply even though it occupies maybe 2% of the image area.

The VAE loss also includes an image-gradient term (L1 on the spatial gradients of the reconstruction vs. target) which encourages sharp edges rather than blurry compromises, plus the standard KL divergence to keep the latent space well-structured.

```
[DIAGRAM: VAE reconstruction comparison. Show 3-4 game frames with their VAE reconstructions
side by side. Include one frame with the bird near a pipe gap to show the bird region is
reconstructed sharply. Possibly show the bird weight mask overlaid on one frame.]
```

## Flow Matching

### Why Flow Matching Over Denoising Diffusion

GameNGen uses denoising diffusion with DDIM sampling, which learns to iteratively remove noise from a corrupted image. Flow matching takes a conceptually different approach: it learns a velocity field that transports samples along straight paths from noise to data.

Concretely, during training we sample a random time t in [0, 1], linearly interpolate between a noise sample z_0 ~ N(0,1) and the target latent z_1 to get z_t = (1-t) * z_0 + t * z_1, and train the model to predict the constant velocity v = z_1 - z_0. At inference, we start from pure noise and integrate this velocity field forward using Euler steps to arrive at the predicted latent.

There are a few reasons this is appealing for our setting. The loss is simple mean-squared error on velocity predictions — no noise schedule to tune, no variance weighting schemes. The ODE paths are straight lines by construction, which means the Euler integrator is already a good approximation even with relatively few steps. And flow matching connects naturally to reflow, a technique for straightening ODE paths further by training on (z_0, z_1) pairs generated by the model itself, enabling even fewer integration steps at inference.

### Euler Integration and the Midpoint Rule

The inference-time ODE is integrated with a simple Euler method: start at z_0, query the model for the velocity at the current position and time, take a step of size dt = 1/N, repeat N times. We use 50 steps, matching the step count used during reflow pair generation in training.

One subtle choice is where to evaluate the velocity within each step. Evaluating at t = i * dt (the left endpoint of each interval) introduces a systematic bias — the final step evaluates at t = 0.98 and the ODE never actually reaches t = 1.0. We use the midpoint rule instead: t = (i + 0.5) * dt. This centers each velocity evaluation within its interval, which both reduces discretization error and makes the forward and backward integrations symmetric — a property that matters for reflow, where we need to reverse the ODE to generate training pairs.

### Reflow for Fewer Steps

Reflow is a two-stage training procedure. In the first stage, the model learns the standard flow matching objective: transport N(0,1) noise to data along straight-line paths. In the second stage, we use this trained model to generate paired (z_0, z_1) samples by taking real data points, integrating backward through the learned ODE to find the corresponding noise points, and then training a new model to connect these specific pairs. Because the backward-then-forward paths are more direct than the original random pairings, the resulting velocity field is "straighter" and can be traversed with fewer Euler steps.

This is still being explored — current training runs use the first-stage objective — but the infrastructure (backward Euler sampling, pair generation) is in place for future experiments targeting 8-step or even single-step inference.

## Conditioning the World Model

The flow model needs to know two things to predict the next frame: what has been happening recently (the visual context), and what the player just did (the action). How these signals enter the model has significant implications for what the model can learn.

### Frame Conditioning via Channel Concatenation

Past frames enter the model through the simplest possible mechanism: channel-wise concatenation. The k most recent latent frames are concatenated with the current noised latent along the channel dimension, producing a tensor with (k+1) * 4 channels that feeds into the U-Net's first convolution. With k = 16 context frames, that's 68 input channels.

This gives the model pixel-aligned access to the past — every spatial position in the input contains the full temporal history at that location. The first convolutional layer learns to mix these channels, effectively learning temporal filters that extract motion, velocity, and change patterns. GameNGen uses the same approach (with 64 context frames), having found it significantly outperforms cross-attention alternatives for frame conditioning.

The context window has evolved over the course of the project. Early experiments used just 2 frames (~67ms of context), which was clearly insufficient — the model had no way to estimate velocities or predict pipe positions beyond the immediate field of view. Increasing to 8 and then 16 frames (roughly half a second at 30fps) gave the model enough temporal context to track scrolling pipes and maintain consistent bird dynamics.

### Action Conditioning via Adaptive Group Normalization

Actions enter the model through a different pathway: learned embeddings injected via Adaptive Group Normalization (AdaGN). Each of the k+1 actions (k context actions plus the current action that should produce the target frame) is mapped through a shared embedding layer, the embeddings are concatenated and projected to a conditioning vector, and this vector modulates the feature maps inside every residual block through learned scale and shift parameters.

This is a deliberate departure from GameNGen, which uses cross-attention for action conditioning — the same mechanism Stable Diffusion uses for text. Cross-attention makes sense when the conditioning signal is a variable-length sequence of rich tokens (like language), but Flappy Bird has a binary action space: flap or don't flap. Encoding 17 binary values (16 context + 1 current) as cross-attention tokens felt like overkill. AdaGN is a lighter mechanism that injects the action signal as a global modulation of every layer's features, which is a natural fit for a low-dimensional discrete control signal.

All three conditioning signals — the flow matching time step t (sinusoidal embedding), the action sequence, and the noise augmentation level — are embedded independently, concatenated, and projected through a shared linear layer to produce the AdaGN scale/shift parameters. This means the model has a single unified conditioning pathway for everything except frame history.

```
[DIAGRAM: Architecture diagram of the ResUNet. Show the U-Net structure with:
- Input: (k+1)*4 channels (concatenated context + noised current)
- Down path with residual blocks and stride-2 convolutions
- Bottleneck with self-attention (144 tokens at 9x16 spatial resolution)
- Up path with skip connections
- AdaGN conditioning injected at every residual block (show the t_emb + c_emb + aug_emb -> scale/shift path)
- Two outputs: velocity field (same spatial dims as input) and done logit (from bottleneck pooling)]
```

### Noise Augmentation for Autoregressive Stability

This is arguably the most important training technique in the entire system. During autoregressive generation, the model conditions on its own previous outputs rather than ground-truth frames. Even small prediction errors accumulate over time, causing the conditioning distribution at inference to drift away from the clean conditioning distribution seen during training. Without mitigation, generation quality collapses within seconds.

The solution, introduced by GameNGen and adapted from cascaded diffusion models, is to deliberately corrupt the conditioning frames during training by adding Gaussian noise at a random intensity. This forces the model to be robust to noisy, imperfect conditioning — exactly what it will encounter during autoregressive inference.

During training, a noise level is sampled uniformly from a set of discrete bins (10 bins spanning standard deviations from 0 to 0.7), Gaussian noise at that intensity is added to the conditioning latents, and the bin index is provided to the model as an additional conditioning signal through a learned embedding. At inference time, we set the augmentation level to zero, telling the model "this conditioning is clean" — but because the model learned to handle noisy conditioning during training, it's naturally robust to the small errors in its own predictions.

The noise augmentation level is discretized rather than continuous because it enters the model through a learned embedding table. This gives the model a clear, discrete signal about how much to trust the conditioning, rather than requiring it to infer the noise level from the data itself.

### Classifier-Free Guidance and the Action-Dropping Problem

Classifier-free guidance (CFG) is a technique that sharpens conditional generation by contrasting the model's conditional and unconditional predictions. During training, the conditioning is randomly dropped (replaced with zeros) for some fraction of samples, teaching the model to make predictions with and without conditioning. At inference, the unconditional prediction is subtracted from the conditional prediction and the difference is amplified: v = v_uncond + w * (v_cond - v_uncond), where w > 1 steers generation more strongly toward the conditioning.

In our context, "conditioning" means the past frame context. GameNGen applies CFG only to frame conditioning, never to actions — the unconditional model still knows what action the player took, it just doesn't know what the game looked like before. This makes conceptual sense: the unconditional prediction should represent "what happens given this action but without visual memory," not "what happens with no information at all."

An earlier version of the training code dropped both frame conditioning and actions during CFG dropout, replacing actions with a null token. This had a subtle but severe effect. The unconditional prediction v_uncond was generated with no knowledge of the action, so CFG amplification conflated "strengthen frame conditioning" with "strengthen action conditioning." In practice, this meant that the CFG-guided prediction could override the action signal — the model would sometimes generate frames where the bird ignored the player's input entirely, or worse, where the bird disappeared. The unconditional model, having no frame context, would predict a generic "average frame" where the bird could be anywhere, and CFG amplification of this against the conditional prediction could destructively interfere with the bird's spatial location.

The fix was straightforward: during CFG dropout, only zero out the frame conditioning; always pass the real actions to both the conditional and unconditional forward passes. This ensures CFG exclusively amplifies the frame-conditioning signal while leaving action conditioning intact.

## Learning to Detect Game Over

A real game engine knows when the game is over because it has explicit collision logic. Our world model has no such logic — it only predicts pixels. But we still need the inference loop to know when to stop, or when to transition to a death sequence.

### The Done Head

We attach a small auxiliary head to the flow model that predicts whether the current frame represents a terminal state. This "done head" is an MLP that reads from the bottleneck features of the U-Net: adaptive average pooling reduces the 9x16 spatial bottleneck to a single vector, which feeds through a linear layer, SiLU activation, and a final linear layer producing a single logit. The binary cross-entropy loss for this head is added to the flow matching loss with a weight of 0.1.

An important design choice: the done head reads from detached bottleneck features. The gradients from the done head objective do not flow back into the encoder or the representation-building layers of the U-Net. This prevents the done prediction task from distorting the learned representations that the flow matching objective depends on — the done head is a passive reader of features, not an active shaper of them.

### Time-Weighted Done Loss

There's a training-inference mismatch in how the done head is used. During training, the done head makes predictions at random noise levels t ~ Uniform(0,1), because the flow matching loss constructs z_t at random interpolation points. But during inference, we only query the done head on the final generated frame — effectively at t = 1, on a clean (fully denoised) sample. This means the done head's predictions at low t values (on heavily noised intermediates) are never actually used.

Rather than only training the done head at t near 1 (which would waste most training signal), we weight the done loss by t^4. This power-law weighting concentrates the learning signal on the clean end of the interpolation while still using all training samples. The weight is normalized to have mean 1 under the uniform distribution on t, so the overall scale of the done loss remains stable: w(t) = 5 * t^4. At t = 0.1, the weight is 0.0005; at t = 0.9, it's 3.28. The model still learns from noisy intermediates, but the gradient signal is dominated by near-clean samples where the done prediction matters.

### Death Frame Semantics

The game environment records extra frames after the terminal collision, showing a game-over overlay sprite. Early experiments recorded 5 such frames, all labeled as terminated. This created several problems: the done head learned to detect the overlay sprite rather than the collision state, the death frames polluted the context window of nearby training samples with post-terminal content that the model would never see during inference, and the class weights for the done loss were distorted by having 6 terminated frames per episode instead of 1.

We settled on recording exactly 1 post-terminal overlay frame, labeled as not-terminated but tagged as post-terminal. The true crash frame (the moment of collision, before the overlay) retains the terminated label. This way, the done head learns to detect the visual signature of collision, and the model can learn the transition from crash to overlay as a normal next-frame prediction. During inference, when the done head fires, we allow one more generation step so the model can produce its own game-over visual, then freeze.

## Fighting Class Imbalance

In Flappy Bird, the optimal strategy involves mostly not flapping. The bird falls under gravity, and flaps are precisely timed, brief interventions. This means the training data is heavily imbalanced: roughly 95% of frames have action = 0 (no flap) and only 5% have action = 1 (flap).

### The Problem: Actions That Don't Matter

With this imbalance, a model can achieve low training loss by essentially ignoring the action input and always predicting "what happens with no flap." The velocity field for action = 1 samples is learned from only 5% of the data, heavily outweighed by the 95% of samples reinforcing the no-flap prediction.

We diagnosed this by running the trained model on the same input with both possible actions and measuring how different the outputs were. If action conditioning is working, feeding action = 0 vs. action = 1 with identical frame context should produce noticeably different predicted frames. In early training runs, this "action diff" metric was around 0.01-0.03 — compared to a typical frame-to-frame difference of 0.17. The model was producing nearly identical outputs regardless of the action.

### Loss Reweighting

The fix is straightforward: weight the flow matching loss for flap samples by the inverse of the class frequency. With a ~17:1 imbalance, flap samples receive 17x the loss weight. This is implemented as a per-sample weight on the mean-squared-error loss, applied before reduction. We use a weighted mean (dividing by the sum of weights rather than the batch size) to keep the loss magnitude stable across batches that happen to have different action distributions.

These weights are computed automatically during the latent encoding step — the encoder counts action = 0 and action = 1 occurrences across the full dataset, computes the ratio, and stores it alongside the encoded data. The flow model training script loads these weights at startup so they always match the current dataset composition. The same approach is used for the done head's positive class weight, since terminal frames are similarly rare (~1-3% of all frames).

## Model Architecture

The flow model is a U-Net with residual blocks, skip connections, and self-attention at the bottleneck. With 128 hidden channels and 2 downsampling layers, the model has approximately 3 million parameters — roughly 300x smaller than the Stable Diffusion U-Net that GameNGen uses. Whether this capacity is sufficient for Flappy Bird is an ongoing question.

### Bottleneck Self-Attention

After two stride-2 downsampling layers, the spatial resolution reaches roughly 9x16 — small enough that full quadratic self-attention over 144 tokens is cheap. We apply multi-head self-attention (8 heads) at this bottleneck, allowing the model to reason about global relationships between spatially distant features. This is particularly relevant for Flappy Bird because the bird's position relative to the pipe gap is the most important piece of game state, and these two elements can be far apart spatially.

The self-attention is applied after the bottleneck residual block and before the done head and upsampling path. The done head reads from post-attention features, so it benefits from the global reasoning that self-attention enables — it can consider the bird-pipe relationship when predicting whether the game is over.

Earlier versions of the model used no attention at all, and an intermediate version experimented with linear attention (using an ELU+1 kernel for O(n) complexity instead of O(n^2)). At 144 tokens, the computational difference between linear and quadratic attention is negligible, so we settled on standard multi-head attention for its well-understood behavior.

### Conditioning Integration

The three conditioning signals — time, actions, and augmentation level — enter every residual block through Adaptive Group Normalization. Inside each block, after the first convolution and group normalization, a linear projection maps the concatenated [t_emb, c_emb, aug_emb] vector (dimension 3 * embed_dim) to a scale and shift of size 2 * fan_out. The scale is applied multiplicatively (as scale + 1 to initialize near identity) and the shift additively.

This projection is zero-initialized, following common practice for conditioning injection: at the start of training, the conditioning has no effect and the model behaves as an unconditional network. As training progresses, the model gradually learns to use the conditioning signals, which stabilizes early training dynamics.

One consequence of cramming all conditioning into a single AdaGN pathway is that time, actions, and augmentation level share the same "bandwidth" into each layer. With a 128-dimensional embedding for each signal, the concatenated vector is 384-dimensional, projected to 2 * fan_out (256 at the first layer, up to 1024 at the bottleneck). For our two-class action space and 10 augmentation bins, this feels like more than enough capacity. It could become a bottleneck for a more complex action space.

```
[DIAGRAM: Detailed view of a single ResBlock showing the AdaGN conditioning path.
Input x -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm (no affine) -> AdaGN modulation
                                                                     ^
                                                           [t_emb | c_emb | aug_emb]
                                                                     |
                                                              Linear (zero-init)
                                                                     |
                                                              (scale, shift)
-> SiLU -> Conv3x3 -> + skip connection -> SiLU -> output]
```

## Inference and Autoregressive Stability

At inference time, the flow model runs in a tight autoregressive loop: accept a keyboard action, build a conditioning tensor from the frame buffer, sample a new latent by Euler-integrating from noise, decode through the VAE, display, and repeat. There are several design choices that affect the stability and quality of this loop.

### Latent Space Clamping

Without any constraint, the Euler integration can push latents arbitrarily far from the training distribution. If the velocity field is slightly off at some point in the ODE trajectory, the latent drifts, the velocity field becomes even less accurate at the new position, and the error compounds. After enough steps, the latent lands in a region the VAE decoder has never seen, producing garbage.

We apply hard clamping at [-4, 4] in normalized latent space after every Euler step. This is a blunt instrument — it introduces discontinuities in the ODE trajectory — but it provides a hard guarantee that latents stay within a reasonable range. In practice, well-behaved generations rarely hit the clamp, and when they do, it typically means the model was already struggling with that particular state.

### Starting Point for Each Frame

Each frame's generation starts from fresh Gaussian noise z_0 ~ N(0,1), independent of the previous frame's latent. This might seem wasteful — why not start from a perturbation of the previous frame, which is presumably close to the target? The reason is that the flow matching objective is trained with z_0 ~ N(0,1), and starting from a different distribution at inference would create a train-test mismatch. The conditioning frames already tell the model what the previous frame looked like; the noise initialization is just the stochastic starting point for the ODE.

That said, this is an area where alternatives could be explored. A perturbation-based initialization (z_0 = z_prev + small noise) would give the model a "warm start" that might reduce integration error, but would require training with a matching initialization distribution. Reflow training naturally produces models that work better with fewer steps from arbitrary starting points, which may make this question less important as the reflow pipeline matures.

## Training at Scale

Training runs on 1-2 NVIDIA H200 GPUs using PyTorch's DistributedDataParallel. Several engineering choices make this tractable.

Mixed-precision training with bfloat16 autocast roughly halves memory usage for activations and speeds up convolutions on Ampere/Hopper GPUs. Gradient checkpointing on every residual block trades compute for memory, re-running forward passes during the backward pass instead of storing all intermediate activations — necessary given the deep U-Net architecture and large batch sizes.

torch.compile with reduce-overhead mode enables CUDA graph capture, eliminating Python overhead in the training loop and allowing kernel fusion. This requires static tensor shapes, which is ensured by dropping the last incomplete batch from each epoch.

The training configuration has evolved substantially. Early runs used batch sizes of 256-512 and 40-65 epochs; the current configuration uses a batch size of 4,096 with 1,000 epochs. The large batch size means each epoch sees relatively few batches (about 16 with the current dataset), so many epochs are needed to see enough gradient updates. A cosine annealing learning rate schedule decays the learning rate from 1e-4 to 1e-6 over the full training run, and gradient clipping at max_norm = 1.0 prevents instability.

## Where Things Stand

The system produces visually sharp frames — the VAE reconstructions are nearly indistinguishable from real game frames, and the flow model generates coherent-looking Flappy Bird scenes. Pipes scroll, the background moves, and the overall visual structure is correct.

However, there are clear remaining issues. The bird sometimes ignores gravity — floating in place rather than falling when the player doesn't flap. The bird occasionally disappears entirely when approaching pipes, likely a lingering artifact of how CFG interacts with spatial predictions. And action responsiveness, while improved by loss reweighting, is still not as crisp as it should be: the difference between flap and no-flap is visible but subtle.

Several hypotheses guide the next steps.

### The Lazy Extrapolation Problem

The most pressing issue is almost certainly action conditioning. The current setup provides 16 frames of dense temporal context — roughly half a second at 30fps. This is enough for the model to estimate the bird's velocity and trajectory with high confidence just by looking at where the bird has been. A single flap, by contrast, contributes a brief upward impulse that changes the trajectory only slightly. The optimization landscape strongly favors learning to extrapolate from temporal context (which explains 95%+ of variance) over learning to attend to the action signal (which explains the remaining few percent).

In other words, dense temporal context gives the model a shortcut: it can get low loss by predicting "where is the bird going based on momentum" and treating the action as noise. AdaGN modulation is a relatively subtle mechanism — it applies a global scale and shift to feature statistics, which the model can learn to partially ignore when the frame context already provides a strong prediction. The result is a model that generates plausible-looking frames but doesn't respond meaningfully to player input.

This creates a fundamental tension in the conditioning design. More context frames help the model understand scene dynamics — pipe positions, scrolling speed, background state. But more context also makes trajectory extrapolation more reliable, weakening the incentive to use the action signal. The action conditioning mechanism needs to be strong enough that ignoring it is costly even when temporal context is rich.

### Spatial Action Conditioning

The most promising architectural change is to make the action signal spatially explicit rather than relying solely on global modulation. One concrete approach: embed the action sequence, tile the embedding spatially, and concatenate it as additional input channels alongside the past frames and noised current frame. This means the first convolutional layer processes the action in the same way it processes pixel-level information — it's architecturally impossible to ignore, because the action is woven into the input tensor at every spatial position.

This could be used alongside AdaGN (which would still carry the action signal into deeper layers) or as a replacement. The key insight is that for a binary action space, the conditioning mechanism needs to be strong relative to the temporal context, not sophisticated. Cross-attention — GameNGen's approach — is designed for rich, variable-length conditioning signals like language. For two possible actions, the challenge isn't representing the action; it's making the model actually use it.

A complementary approach is to reduce the context window (from 16 to 4-8 frames) to weaken the extrapolation shortcut, forcing greater reliance on the action. This trades temporal context for action sensitivity. The right balance likely depends on the specific game — Flappy Bird's dynamics are simple enough that 4 frames may provide sufficient velocity information, while the reduced context forces the model to attend to the action to predict what happens next.

### Other Directions

**More data, different data.** 470K frames is workable but thin. Increasing data by 5-10x, with even more aggressive behavioral diversification, would give the model more examples of rare but important states — especially bird-pipe interactions and extreme heights.

**Decoder fine-tuning.** GameNGen fine-tunes the VAE decoder separately on generated latents, using pixel-space MSE loss against real frames. This corrects for the distribution shift between VAE-encoded latents (used during training) and flow-model-generated latents (used at inference). We haven't implemented this yet, but it's a natural next step once the flow model is generating plausible content.

**Reflow for faster inference.** The current 50-step Euler integration takes ~70ms per frame, limiting inference to roughly 10 FPS. Reflow training could reduce this to 8 or fewer steps, enabling real-time 30 FPS generation. The infrastructure exists; it's a matter of training the second-stage model once the first stage is stable.

**Model scale.** The current model is around 3M parameters — roughly 300x smaller than GameNGen's Stable Diffusion backbone, but likely more than Flappy Bird requires. DIAMOND, a related world model for Atari, uses 4.4M parameters across 26 games, many more visually complex than Flappy Bird. There's likely room to shrink the model once the conditioning issues are resolved, which would also speed up inference. But model size isn't the bottleneck right now — a too-large model that ignores actions will still ignore actions when it's smaller. The architecture of how information flows matters more than the capacity of the network.

This project is a work in progress. The architecture is in place, the training pipeline is functional, and the model is generating recognizable Flappy Bird gameplay. The gap between "recognizable" and "convincing" turns out to be where all the interesting design questions live — and they are questions about inductive biases, not scale. How does the structure of the conditioning mechanism shape what the model learns to rely on? When does providing more information actually hurt by enabling shortcuts? How strong does a signal need to be before the model will use it over a simpler alternative? These are the questions that make this kind of project interesting beyond the immediate application.
