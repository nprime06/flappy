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
from diffuse.nn.ae import VAE
from diffuse.nn.resunet import ResUNet
from diffuse.ngen.sampler import euler_sample, euler_sample_cfg


def load_vae(checkpoint_path, device):
    """Load frozen VAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    vae = VAE(
        image_channels=cfg["image_channels"],
        hidden_channels=cfg["hidden_channels"],
        latent_channels=cfg["latent_channels"],
        num_layers=cfg["num_layers"],
    ).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_flow_model(checkpoint_path, device):
    """Load flow model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = ResUNet(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        embed_dim=cfg["embed_dim"],
        num_classes=cfg["num_classes"],
        context_channels=cfg["context_frames"] * cfg["in_channels"],
        num_aug_bins=cfg["num_aug_bins"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
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
    with torch.no_grad():
        x = vae_decode(vae, z, latent_mean, latent_std)
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        x = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return x


def main():
    parser = argparse.ArgumentParser(description="Play Flappy Bird through the world model")
    parser.add_argument("--ngen-checkpoint", type=str, required=True,
                        help="Path to flow model checkpoint")
    parser.add_argument("--vae-checkpoint", type=str, default=None,
                        help="Path to VAE checkpoint (reads from ngen config if not specified)")
    parser.add_argument("--vod-dir", type=str, default=None,
                        help="Path to vod directory for initial frames")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of Euler steps for sampling (match training)")
    parser.add_argument("--scale", type=int, default=2,
                        help="Display scale factor")
    parser.add_argument("--noise-scale", type=float, default=0.1,
                        help="Scale of noise perturbation for z_0 (0 = pure noise, small = perturbation)")
    parser.add_argument("--aug-level", type=int, default=8,
                        help="Augmentation level at inference (0-15, 8 = median)")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="Temporal smoothing factor (0 = none, >0 blends with previous)")
    parser.add_argument("--done-threshold", type=float, default=0.5,
                        help="Threshold for done prediction to end game")
    parser.add_argument("--use-cfg", action="store_true",
                        help="Use classifier-free guidance during sampling")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="CFG scale (only used with --use-cfg)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load flow model and config
    print(f"Loading flow model from {args.ngen_checkpoint}")
    flow_model, model_cfg = load_flow_model(args.ngen_checkpoint, device)

    # Load training config from checkpoint
    ckpt = torch.load(args.ngen_checkpoint, map_location=device, weights_only=False)
    train_cfg = ckpt["train_config"]

    # Get VAE checkpoint path
    vae_path = args.vae_checkpoint or train_cfg["vae_checkpoint"]
    # Handle cluster vs local paths
    if not os.path.exists(vae_path):
        # Try local path
        local_vae = vae_path.replace("/home/willzhao/flappy/diffuse",
                                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if os.path.exists(local_vae):
            vae_path = local_vae
        else:
            print(f"VAE checkpoint not found at {vae_path}")
            print("Please specify --vae-checkpoint")
            sys.exit(1)

    print(f"Loading VAE from {vae_path}")
    vae = load_vae(vae_path, device)

    # Get vod directory
    vod_dir = args.vod_dir or train_cfg.get("data_dir")
    if vod_dir and not os.path.exists(vod_dir):
        # Try local path
        local_vod = vod_dir.replace("/home/willzhao/flappy",
                                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if os.path.exists(local_vod):
            vod_dir = local_vod

    if not vod_dir or not os.path.exists(vod_dir):
        print(f"VOD directory not found. Please specify --vod-dir")
        sys.exit(1)

    # Config values
    k = model_cfg["context_frames"]
    latent_mean = train_cfg["latent_mean"]
    latent_std = train_cfg["latent_std"]
    num_aug_bins = model_cfg["num_aug_bins"]

    print(f"Context frames: {k}")
    print(f"Latent normalization: mean={latent_mean:.4f}, std={latent_std:.4f}")

    # Load initial frames
    print(f"Loading initial frames from {vod_dir}")
    past_frames, current_frame = load_initial_frames(vod_dir, k, device)

    # Encode initial frames
    with torch.no_grad():
        B = 1
        past_flat = past_frames.flatten(0, 1)  # (k, C, H, W)
        z_cond = vae_encode(vae, past_flat, latent_mean, latent_std)  # (k, latent_ch, H', W')
        z_cond = z_cond.unsqueeze(0).flatten(1, 2)  # (1, k*latent_ch, H', W')

        z_current = vae_encode(vae, current_frame, latent_mean, latent_std)

    # Get image dimensions from current frame
    H, W = current_frame.shape[2], current_frame.shape[3]

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((W * args.scale, H * args.scale))
    pygame.display.set_caption("Flappy Bird - World Model")
    clock = pygame.time.Clock()

    # Frame buffer: list of k latents for conditioning
    frame_buffer = [z_cond[:, i*model_cfg["in_channels"]:(i+1)*model_cfg["in_channels"]]
                    for i in range(k)]

    # Current latent (the frame we display)
    z_display = z_current

    # aug_level: use median level at inference for stability
    aug_level = torch.full((1,), args.aug_level, dtype=torch.long, device=device)

    running = True
    frame_count = 0
    start_time = time.time()

    print("\nControls:")
    print("  SPACE - Flap")
    print("  Q/ESC - Quit")
    print("\nStarting inference loop...")

    while running:
        # Handle events
        action = 0  # no flap by default
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # flap
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Also check if space is held
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action = 1

        # Prepare conditioning
        z_cond = torch.cat(frame_buffer, dim=1)  # (1, k*latent_ch, H', W')
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)

        # Sample next frame using flow model
        with torch.no_grad():
            # Use perturbation from current frame instead of pure noise
            if args.noise_scale > 0:
                z_0 = z_display + args.noise_scale * torch.randn_like(z_display)
            else:
                z_0 = torch.randn_like(z_display)

            # Use CFG if enabled
            if args.use_cfg:
                z_next = euler_sample_cfg(
                    flow_model, z_0, z_cond, action_tensor, aug_level,
                    cfg_scale=args.cfg_scale, num_steps=args.num_steps
                )
            else:
                z_next = euler_sample(
                    flow_model, z_0, z_cond, action_tensor, aug_level,
                    num_steps=args.num_steps
                )

            # Apply temporal smoothing if enabled
            if args.smoothing > 0:
                z_next = (1 - args.smoothing) * z_next + args.smoothing * z_display

            # Check done prediction if model has done head
            if hasattr(flow_model, 'use_done_head') and flow_model.use_done_head:
                t_final = torch.ones(1, device=device)
                _, done_logit = flow_model(z_next, t_final, c=action_tensor, z_cond=z_cond,
                                           aug_level=aug_level, return_done=True)
                done_prob = torch.sigmoid(done_logit).item()
                if done_prob > args.done_threshold:
                    print(f"Game Over! (done_prob={done_prob:.2f})")
                    # Could break here or reset to new episode
                    # For now, just print and continue

        # Update frame buffer (shift left, add new)
        frame_buffer.pop(0)
        frame_buffer.append(z_display.clone())

        # Update current display latent
        z_display = z_next

        # Decode and display
        img = latent_to_image(vae, z_display, latent_mean, latent_std)

        # Convert to pygame surface
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (W * args.scale, H * args.scale))
        screen.blit(surf, (0, 0))

        # Show action indicator
        if action == 1:
            pygame.draw.circle(screen, (255, 255, 0), (20, 20), 10)

        pygame.display.flip()

        # Frame rate control
        clock.tick(30)
        frame_count += 1

        # FPS counter
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            pygame.display.set_caption(f"Flappy Bird - World Model | FPS: {fps:.1f}")

    pygame.quit()
    print(f"\nGenerated {frame_count} frames in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
