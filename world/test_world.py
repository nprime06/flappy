"""Interactive world model inference - play Flappy Bird through the neural network."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cache for the game over sprite
_gameover_sprite = None

def _load_gameover_sprite():
    """Load and cache the game over sprite from flappy_bird_gymnasium assets."""
    global _gameover_sprite
    if _gameover_sprite is None:
        import flappy_bird_gymnasium
        assets_dir = Path(flappy_bird_gymnasium.__file__).parent / "assets" / "sprites"
        gameover_path = assets_dir / "gameover.png"
        _gameover_sprite = Image.open(gameover_path).convert("RGBA")
    return _gameover_sprite


def _add_gameover_overlay(img: np.ndarray) -> np.ndarray:
    """Add the game over sprite overlay to an image."""
    gameover = _load_gameover_sprite()
    frame_pil = Image.fromarray(img).convert("RGBA")
    x = (frame_pil.width - gameover.width) // 2
    y = int(frame_pil.height * 0.4)
    frame_pil.paste(gameover, (x, y), gameover)
    return np.array(frame_pil.convert("RGB"))
from diffuse.nn.vae import VAE
from diffuse.nn.resunet import ResUNet
from diffuse.ngen.sampler import euler_sample

# Fallback values (for backward compatibility)
DEFAULT_LATENT_MEAN = 0.4735
DEFAULT_LATENT_STD = 1.5931


def load_vae(checkpoint_path, device):
    """Load frozen VAE from checkpoint and return latent statistics.
    
    Returns:
        tuple of (vae, latent_mean, latent_std)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    vae = VAE(
        image_channels=cfg["image_channels"],
        hidden_channels=cfg["hidden_channels"],
        latent_channels=cfg["latent_channels"],
        num_layers=cfg["num_layers"],
    ).to(device)
    state_dict = ckpt["model"]
    
    # handle compiled model checkpoints (_orig_mod prefix)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
    
    vae.load_state_dict(state_dict)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    
    # Load latent statistics from checkpoint
    latent_mean = ckpt.get("latent_mean", DEFAULT_LATENT_MEAN)
    latent_std = ckpt.get("latent_std", DEFAULT_LATENT_STD)
    
    if "latent_mean" in ckpt and "latent_std" in ckpt:
        print(f"Loaded latent statistics from checkpoint: mean={latent_mean:.4f}, std={latent_std:.4f}")
    else:
        print(f"Warning: checkpoint does not contain latent statistics, using defaults")
    
    return vae, latent_mean, latent_std


def load_flow_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = ResUNet(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        embed_dim=cfg["embed_dim"],
        act_embed_dim=cfg["act_embed_dim"],
        num_classes=cfg["num_classes"],
        context_size=cfg["context_size"],
        num_aug_bins=cfg["num_aug_bins"],
    ).to(device)
    state_dict = ckpt["model"]
    
    # handle compiled model checkpoints (_orig_mod prefix)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


def load_initial_frames(vod_dir, k, device):
    """Load k initial frames from a random vod run."""
    vod_path = Path(vod_dir)

    # Find a category and run
    categories = [d for d in vod_path.iterdir() if d.is_dir()]
    if not categories:
        raise ValueError(f"No categories found in {vod_dir}")

    category = categories[0]
    runs = [d for d in category.iterdir() if d.is_dir()]
    if not runs:
        raise ValueError(f"No runs found in {category}")

    run_dir = runs[0]
    frames_dir = run_dir / "frames"

    # Get first k+1 frames (we need k context + 1 current)
    frame_files = sorted(frames_dir.glob("*.png"))[:k+1]
    if len(frame_files) < k + 1:
        raise ValueError(f"Not enough frames in {frames_dir}")

    frames = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor * 2 - 1
        frames.append(img_tensor)

    # Stack: past k frames and current frame
    past_frames = torch.stack(frames[:k]).unsqueeze(0).to(device)  # (1, k, C, H, W)
    current_frame = frames[k].unsqueeze(0).to(device)  # (1, C, H, W)

    return past_frames, current_frame


def vae_encode(vae, x, latent_mean, latent_std):
    """Encode image to normalized latent mean (deterministic)."""
    encoded = vae.encoder(x)
    z, _, _ = vae.reparameterize(encoded, sample=False)
    z = (z - latent_mean) / latent_std
    return z


def vae_decode(vae, z, latent_mean, latent_std):
    """Decode normalized latent back to image."""
    z = z * latent_std + latent_mean
    return vae.decoder(z)


def latent_to_image(vae, z, latent_mean, latent_std):
    """Convert latent to displayable image."""
    with torch.inference_mode():
        x = vae_decode(vae, z, latent_mean, latent_std)
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        x = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return x


def load_random_start_latent(latent_vod_dir, k, device):
    """Load k context frames + current frame from a random position in a random episode."""
    import json, random

    # collect all valid runs
    runs = []
    for run_dir in Path(latent_vod_dir).glob("*/*/"):
        latents_file = run_dir / "latents.pt"
        info_file = run_dir / "run_info.jsonl"
        if latents_file.exists() and info_file.exists():
            runs.append(run_dir)

    if not runs:
        raise ValueError(f"No valid runs found in {latent_vod_dir}")

    run_dir = random.choice(runs)
    latents = torch.load(run_dir / "latents.pt", map_location=device, weights_only=True)
    n_frames = latents.shape[0]

    actions_dict = {}
    with open(run_dir / "run_info.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if "step" in rec:
                actions_dict[rec["step"]] = rec["action"]

    # pick a random valid starting position (need k context + 1 current)
    start = random.randint(k, n_frames - 1)
    past_latents = latents[start - k:start].unsqueeze(0).to(device)  # (1, k, C, H', W')
    current_latent = latents[start].unsqueeze(0).to(device)  # (1, C, H', W')

    # load k+1 actions: context actions + target action
    action_list = [actions_dict.get(i, 0) for i in range(start - k, start + 1)]

    print(f"Random start: run={run_dir.name}, frame={start}/{n_frames}")
    print(f"  Actions loaded: {action_list}")
    return past_latents, current_latent, action_list


def main():
    parser = argparse.ArgumentParser(description="Play Flappy Bird through the world model")
    parser.add_argument("--ngen-checkpoint", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--vod-dir", type=str, required=True)
    parser.add_argument("--latent-vod", type=str, default=None)
    parser.add_argument("--random-start", action="store_true", help="start from random VOD position (requires --latent-vod)")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--done-threshold", type=float, default=0.5)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--debug", action="store_true") # print debug info every frame
    args = parser.parse_args()

    if args.random_start and not args.latent_vod:
        print("Error: --random-start requires --latent-vod")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load flow model and config
    print(f"Loading flow model from {args.ngen_checkpoint}")
    flow_model, model_cfg = load_flow_model(args.ngen_checkpoint, device)

    # Get VAE checkpoint path (required)
    if not args.vae_checkpoint:
        print("Error: --vae-checkpoint is required")
        sys.exit(1)
    vae_path = args.vae_checkpoint

    print(f"Loading VAE from {vae_path}")
    vae, latent_mean, latent_std = load_vae(vae_path, device)

    # Get vod directory (required)
    if not args.vod_dir:
        print("Error: --vod-dir is required")
        sys.exit(1)
    vod_dir = args.vod_dir

    # Config values
    k = model_cfg["context_size"]
    num_aug_bins = model_cfg["num_aug_bins"]

    print(f"Context frames: {k}")
    print(f"Latent normalization: mean={latent_mean:.4f}, std={latent_std:.4f}")

    # Load initial frames
    if args.random_start:
        print(f"Loading random start from latent VOD: {args.latent_vod}")
        past_latents, z_current, init_actions = load_random_start_latent(args.latent_vod, k, device)
        # past_latents: (1, k, C, H', W'), already in latent space
        z_cond = past_latents.flatten(1, 2)  # (1, k*latent_ch, H', W')
        # load a pixel frame just to get display dimensions
        past_frames, _ = load_initial_frames(vod_dir, k, device)
        H, W = past_frames.shape[3], past_frames.shape[4]
    else:
        init_actions = None
        print(f"Loading initial frames from {vod_dir}")
        past_frames, current_frame = load_initial_frames(vod_dir, k, device)
        with torch.inference_mode():
            past_flat = past_frames.flatten(0, 1)  # (k, C, H, W)
            z_cond = vae_encode(vae, past_flat, latent_mean, latent_std)  # (k, latent_ch, H', W')
            z_cond = z_cond.unsqueeze(0).flatten(1, 2)  # (1, k*latent_ch, H', W')
            z_current = vae_encode(vae, current_frame, latent_mean, latent_std)
        H, W = current_frame.shape[2], current_frame.shape[3]

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((W * 2, H * 2))
    pygame.display.set_caption("Flappy Bird - World Model")
    clock = pygame.time.Clock()

    # Frame buffer: list of k latents for conditioning
    # Initially contains frames[0] through frames[k-1] from the VOD
    frame_buffer = [z_cond[:, i*model_cfg["in_channels"]:(i+1)*model_cfg["in_channels"]]
                    for i in range(k)]

    # Action buffer: list of k past actions for conditioning
    if init_actions is not None:
        action_buffer = init_actions[:k]  # first k actions from VOD (real context actions)
    else:
        action_buffer = [0] * k  # dummy actions when starting from pixel frames

    # Current latent (the frame we display)
    # Initially this is frame[k] from the VOD
    z_display = z_current
    
    print(f"\nInitial state (loaded from VOD):")
    print(f"  frame_buffer: VOD frames[0] to VOD frames[{k-1}] (k={k} context frames)")
    print(f"  action_buffer: [0, 0, ..., 0] (k={k} dummy actions, we don't know real VOD actions)")
    print(f"  z_display: VOD frame[{k}] (will be displayed, then used in next prediction)")
    print(f"\nLoop iteration frame_count=0 will:")
    print(f"  - Display VOD frame[{k}]")
    print(f"  - Get user action")
    print(f"  - Predict first generated frame using VOD frames[0:{k}] + user action")

    aug_level = torch.full((1,), 0, dtype=torch.long, device=device) 

    running = True
    frame_count = 0
    start_time = time.time()

    print("\nControls:")
    print("  SPACE - Flap")
    print("  D     - Toggle death mode (red tint preview)")
    print("  Q/ESC - Quit")
    print("\nStarting inference loop...")

    manual_death_mode = False  # For testing red tint
    # When the done head fires, we allow the model to generate N more frames, then freeze.
    # This lets the world model itself generate a single "game over" frame if it has learned it.
    stop_countdown: int | None = None

    while running:
        # ============================================================================
        # ACTION-FRAME ALIGNMENT EXPLANATION
        # ============================================================================
        # At the start of iteration for frame_count=t:
        #   - z_display = frame[t-1] (what we just displayed)
        #   - frame_buffer = [frame[t-k-1], ..., frame[t-2]]  (k old context frames)
        #   - action_buffer = [action[t-k-1], ..., action[t-2]]  (k old actions)
        #
        # We will:
        #   1. Get action[t-1] from user input (action taken NOW at frame[t-1])
        #   2. Update frame_buffer: pop oldest, append frame[t-1] (z_display)
        #      -> frame_buffer = [frame[t-k], ..., frame[t-1]]
        #   3. Build actions: [action[t-k], ..., action[t-1]] (k+1 actions total)
        #      - First k actions caused the k context frames
        #      - Last action (action[t-1]) will cause frame[t] (the target)
        #   4. Predict frame[t] using context frames[t-k:t] and actions[t-k:t]
        #   5. Update: z_display = frame[t], action_buffer = [action[t-k], ..., action[t-1]]
        #
        # Key: action[i] taken at frame[i] causes transition to frame[i+1]
        # ============================================================================
        
        # Handle events
        action = 0  # no flap by default
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # flap
                elif event.key == pygame.K_d:
                    manual_death_mode = not manual_death_mode
                    print(f"\nDeath mode: {'ON' if manual_death_mode else 'OFF'}")
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Also check if space is held
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action = 1

        # Update frame buffer FIRST to include current displayed frame in context
        # This is critical: we predict the NEXT frame, so context must include current frame
        frame_buffer.pop(0)
        frame_buffer.append(z_display.clone())

        # Prepare conditioning (now includes z_display)
        z_cond = torch.cat(frame_buffer, dim=1)  # (1, k*latent_ch, H', W')

        # Build actions tensor: k actions that caused context frames + action causing target
        # action_buffer contains k actions for context, current action is for target
        actions = torch.tensor([action_buffer + [action]], dtype=torch.long, device=device)  # (1, k+1)
        
        # ACTION ALIGNMENT VERIFICATION
        # Log the action-frame relationship every frame
        if frame_count % 30 == 0 or frame_count < 10:
            print(f"\n{'='*80}")
            print(f"PREDICTING FRAME {frame_count} - ACTION ALIGNMENT CHECK")
            print(f"{'='*80}")
            print(f"Context frames in frame_buffer (after update):")
            print(f"  frame_buffer contains: frames[{frame_count-k}] through frames[{frame_count-1}]")
            print(f"  (k={k} frames used as conditioning)")
            print(f"\nAction-Frame causality mapping:")
            for i in range(k):
                frame_idx = frame_count - k + i
                act = action_buffer[i]
                next_frame_idx = frame_idx + 1
                print(f"  actions[{i}] = {act} (action at frame[{frame_idx}]) -> caused frame[{next_frame_idx}]")
            print(f"  actions[{k}] = {action} (action at frame[{frame_count-1}]) -> will cause frame[{frame_count}] (TARGET)")
            print(f"\nActions tensor passed to model: {actions[0].tolist()}")
            print(f"  Shape: {actions.shape} = (batch_size=1, k+1={k+1})")
            print(f"\nConditioning tensors:")
            print(f"  z_cond.shape: {z_cond.shape} = (1, k*latent_ch={k*model_cfg['in_channels']}, H, W)")
            print(f"  z_0.shape (noise): will be {z_display.shape}")
            print(f"\nTarget to predict: frame[{frame_count}]")
            print(f"{'='*80}\n")

        # Sample next frame using flow model
        with torch.inference_mode():
            z_0 = torch.randn_like(z_display)
            ngen_start = time.perf_counter()
            z_next = euler_sample(
                    flow_model, z_0, z_cond, actions, aug_level,
                    cfg_scale=args.cfg_scale, num_steps=args.num_steps
                )

            # Check done prediction (model always returns done_logit)
            t_final = torch.ones(1, device=device)
            _, done_logit = flow_model(z_next, t_final, z_cond=z_cond, c=actions, aug_level=aug_level)
            done_prob = torch.sigmoid(done_logit).item()
            ngen_elapsed = time.perf_counter() - ngen_start
            
            # Log alignment verification after prediction
            if frame_count % 30 == 0 or frame_count < 10:
                print(f"✓ PREDICTION COMPLETE for frame[{frame_count}]:")
                print(f"  Context used: frames[{frame_count-k}] to frames[{frame_count-1}]")
                print(f"  Actions used: actions[{frame_count-k}] to actions[{frame_count-1}]")
                print(f"  Key: action[{frame_count-1}]={action} caused frame[{frame_count}]")
                print(f"  Done probability: {done_prob:.3f}")
                print(f"  Timing: ngen={ngen_elapsed:.3f}s")
                print()
            else:
                print(f"Frame {frame_count}: done_prob={done_prob:.3f}", end="\r")
            
            if done_prob > args.done_threshold:
                if stop_countdown is None:
                    # Include the current frame + one additional generated frame.
                    stop_countdown = 2
                    print(f"\nGame Over detected (done_prob={done_prob:.2f}, frame={frame_count}). "
                          f"Generating one more frame then freezing...")

        # Debug logging
        if args.debug and frame_count % 10 == 0:
            # Test action conditioning: what would happen with opposite action?
            with torch.inference_mode():
                alt_action = 1 - action
                alt_actions = torch.tensor([action_buffer + [alt_action]], dtype=torch.long, device=device)
                z_next_alt = euler_sample(
                    flow_model, z_0, z_cond, alt_actions, aug_level,
                    cfg_scale=args.cfg_scale, num_steps=args.num_steps
                )
                action_diff = torch.abs(z_next - z_next_alt).mean().item()

            print(f"Frame {frame_count}: action={action}, "
                  f"z_cond=[{z_cond.min():.2f}, {z_cond.max():.2f}], "
                  f"z_next=[{z_next.min():.2f}, {z_next.max():.2f}], "
                  f"frame_diff={torch.abs(z_next - z_display).mean():.4f}, "
                  f"action_diff={action_diff:.4f}")

        # Update current display latent and action buffer
        z_display = z_next
        action_buffer.pop(0)
        action_buffer.append(action)
        
        # Verify state after update (for next iteration)
        if (frame_count % 30 == 0 or frame_count < 10) and frame_count < 5:
            print(f"State after update (ready for frame[{frame_count+1}]):")
            print(f"  z_display: now frame[{frame_count}]")
            print(f"  action_buffer: {action_buffer} (actions[{frame_count-k+1}] to actions[{frame_count}])")
            print()

        # Decode and display
        decode_start = time.perf_counter()
        img = latent_to_image(vae, z_display, latent_mean, latent_std)
        decode_elapsed = time.perf_counter() - decode_start
        print(
            f"Frame {frame_count}: ngen={ngen_elapsed:.3f}s, decode={decode_elapsed:.3f}s"
        )

        # IMPORTANT: no external compositing for "game over" — pixels should come from the model.

        # Convert to pygame surface
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (W * 2, H * 2))
        screen.blit(surf, (0, 0))

        # Show action indicator
        if action == 1:
            pygame.draw.circle(screen, (255, 255, 0), (20, 20), 10)

        pygame.display.flip()

        # Frame rate control
        clock.tick(30)
        frame_count += 1

        if stop_countdown is not None:
            stop_countdown -= 1
            if stop_countdown <= 0:
                running = False

        # FPS counter
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            pygame.display.set_caption(f"Flappy Bird - World Model | FPS: {fps:.1f}")

    pygame.quit()
    print(f"\nGenerated {frame_count} frames in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
