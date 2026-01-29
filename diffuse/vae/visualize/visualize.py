import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nn.vae import VAE

# Get the directory of this file for default output paths
VISUALIZE_DIR = Path(__file__).parent

def load_random_frame(vod_dir):
    frame_paths = sorted(Path(vod_dir).glob("*/*/frames/*.png"))
    path = random.choice(frame_paths)
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img)  # [0, 1]
    tensor = tensor * 2 - 1  # [-1, 1]
    return tensor, path


def tensor_to_display(tensor):
    """Convert tensor from [-1, 1] to numpy array for display."""
    img = (tensor + 1) / 2  # back to [0, 1]
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def visualize_downsample(vod_dir, scale_factor=0.5, modes=("area", "bilinear"), 
                          output_path=None, num_frames=30, fps=30):
    """Compare original frames with downsampled versions using different modes and create a GIF."""
    if output_path is None:
        output_path = VISUALIZE_DIR / "downsample.gif"
    output_path = str(output_path)  # Convert Path to string for imageio

    # Find all episode directories and pick a random one
    episode_dirs = sorted(Path(vod_dir).glob("*/*/frames"))
    episode_dir = random.choice(episode_dirs)
    frame_paths = sorted(episode_dir.glob("*.png"))

    # Pick a random starting point that allows num_frames contiguous frames
    max_start = max(0, len(frame_paths) - num_frames)
    start_idx = random.randint(0, max_start)
    frame_paths = frame_paths[start_idx:start_idx + num_frames]

    print(f"Episode: {episode_dir}")
    print(f"Frames {start_idx} to {start_idx + len(frame_paths) - 1}")

    # Load frames as batch
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        tensor = transforms.ToTensor()(img) * 2 - 1  # [-1, 1]
        frames.append(tensor)
    batch = torch.stack(frames)

    print(f"Original shape: {batch.shape}")

    # Create GIF frames showing original and downsampled versions side by side
    gif_frames = []
    n_modes = len(modes)
    
    for i in range(len(frames)):
        frame_tensor = batch[i]
        frame_batch = frame_tensor.unsqueeze(0)
        
        # Get original frame for display
        orig_display = tensor_to_display(frame_tensor)
        
        # Create downsampled versions
        downsampled_displays = []
        for mode in modes:
            downsampled = F.interpolate(frame_batch, scale_factor=scale_factor, mode=mode)
            downsampled = downsampled.squeeze(0)
            # Upsample back to original size for side-by-side comparison
            downsampled_upsampled = F.interpolate(
                downsampled.unsqueeze(0), 
                size=frame_tensor.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            downsampled_displays.append(tensor_to_display(downsampled_upsampled))
            if i == 0:  # Print shape info only for first frame
                print(f"{mode} shape: {downsampled.shape}")
        
        # Concatenate original and all downsampled versions horizontally
        combined = np.concatenate([orig_display] + downsampled_displays, axis=1)
        # Convert to uint8
        combined = (combined * 255).astype(np.uint8)
        gif_frames.append(combined)

    # Save GIF
    imageio.mimsave(output_path, gif_frames, fps=fps)
    print(f"Saved GIF to: {output_path}")

    # Also show first frame comparison
    n_cols = 1 + len(modes)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # Original
    axes[0].imshow(tensor_to_display(batch[0]))
    axes[0].set_title(f"Original\n{batch.shape[2]}x{batch.shape[3]}")
    axes[0].axis("off")
    
    # Downsampled versions (upsampled for display)
    frame_batch = batch[0:1]
    for i, mode in enumerate(modes):
        downsampled = F.interpolate(frame_batch, scale_factor=scale_factor, mode=mode)
        downsampled_upsampled = F.interpolate(
            downsampled, 
            size=batch.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        axes[i + 1].imshow(tensor_to_display(downsampled_upsampled[0]))
        axes[i + 1].set_title(f"{mode}\n{downsampled.shape[2]}x{downsampled.shape[3]}")
        axes[i + 1].axis("off")
    
    plt.tight_layout()
    plt.show()


def visualize_reconstruction(checkpoint_path, vod_dir, output_path=None,
                              num_frames=30, fps=30, device="cpu"):
    """Load a VAE checkpoint, encode/decode contiguous frames, and create side-by-side GIF."""
    if output_path is None:
        output_path = VISUALIZE_DIR / "reconstruction.gif"
    output_path = str(output_path)  # Convert Path to string for imageio
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint or use defaults
    config = ckpt.get("model_config", {})
    model = VAE(
        image_channels=config.get("image_channels", 3),
        hidden_channels=config.get("hidden_channels", 16),
        latent_channels=config.get("latent_channels", 4),
        num_layers=config.get("num_layers", 4),
        decoder_hidden_channels=config.get("decoder_hidden_channels"),
        decoder_num_layers=config.get("decoder_num_layers"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Epoch: {ckpt.get('epoch', 'unknown')}")

    # Find all episode directories and pick a random one
    episode_dirs = sorted(Path(vod_dir).glob("*/*/frames"))
    episode_dir = random.choice(episode_dirs)
    frame_paths = sorted(episode_dir.glob("*.png"))

    # Pick a random starting point that allows num_frames contiguous frames
    max_start = max(0, len(frame_paths) - num_frames)
    start_idx = random.randint(0, max_start)
    frame_paths = frame_paths[start_idx:start_idx + num_frames]

    print(f"Episode: {episode_dir}")
    print(f"Frames {start_idx} to {start_idx + len(frame_paths) - 1}")

    # Load frames as batch
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        tensor = transforms.ToTensor()(img) * 2 - 1  # [-1, 1]
        frames.append(tensor)
    batch = torch.stack(frames).to(device)

    # Encode and decode
    with torch.no_grad():
        encoded = model.encoder(batch)
        z, mean, logvar = model.reparameterize(encoded, sample=False)
        recon = model.decoder(z).cpu()

    # If reconstruction is different resolution, upsample for display
    if recon.shape[-2:] != batch.shape[-2:]:
        recon = F.interpolate(recon, size=batch.shape[-2:], mode='bilinear', align_corners=False)

    batch = batch.cpu()

    # Compute average L1
    l1 = F.l1_loss(recon, batch).item()
    print(f"Average L1: {l1:.4f}")

    # Create side-by-side frames for GIF
    gif_frames = []
    for i in range(len(frames)):
        orig = tensor_to_display(batch[i])
        rec = tensor_to_display(recon[i])

        # Concatenate side by side
        combined = np.concatenate([orig, rec], axis=1)
        # Convert to uint8
        combined = (combined * 255).astype(np.uint8)
        gif_frames.append(combined)

    # Save GIF
    imageio.mimsave(output_path, gif_frames, fps=fps)
    print(f"Saved GIF to: {output_path}")

    # Also show first frame comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(tensor_to_display(batch[0]))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(tensor_to_display(recon[0]))
    axes[1].set_title(f"Reconstruction (L1: {l1:.4f})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

def visualize_detection(vod_dir, output_path=None, num_frames=60, fps=30):
    """Create a GIF showing bird bounding box detection on video frames.
    
    The bird has a fixed x position (world scrolls past it), so we only
    detect the top edge (y_min) and use fixed x bounds and height.
    """
    if output_path is None:
        output_path = VISUALIZE_DIR / "bird_detection.gif"
    output_path = str(output_path)  # Convert Path to string for imageio

    # Find all episode directories and pick a random one
    episode_dirs = sorted(Path(vod_dir).glob("*/*/frames"))
    episode_dir = random.choice(episode_dirs)
    frame_paths = sorted(episode_dir.glob("*.png"))

    # Pick a random starting point
    max_start = max(0, len(frame_paths) - num_frames)
    start_idx = random.randint(0, max_start)
    frame_paths = frame_paths[start_idx:start_idx + num_frames]

    print(f"Episode: {episode_dir}")
    print(f"Frames {start_idx} to {start_idx + len(frame_paths) - 1}")

    transform = transforms.ToTensor()
    gif_frames = []
    detected_count = 0

    # Hardcoded bird detection parameters
    bird_x_min = 50
    bird_x_max = 100
    bird_height = 35

    for path in frame_paths:
        # Load and convert to tensor
        img_pil = Image.open(path).convert("RGB")
        img_tensor = transform(img_pil) * 2 - 1  # [-1, 1]

        # Detect bird bbox (inline detection logic)
        img = (img_tensor + 1) / 2  # [0, 1]
        r, g, b = img[0], img[1], img[2]

        # Detect orange bird body
        bird_mask = (r > 0.8) & (g > 0.3) & (g < 0.6) & (b < 0.3)

        bbox = None
        if bird_mask.any():
            H, W = bird_mask.shape

            # Find top edge of bird (y_min)
            rows = bird_mask.any(dim=1).nonzero(as_tuple=True)[0]
            y_min = rows.min().item() - 23  # Hardcoded offset

            # Fixed x bounds and height
            y_max = min(H - 1, y_min + bird_height)

            bbox = (bird_x_min, y_min, bird_x_max, y_max)

        # Draw bbox on image
        draw = ImageDraw.Draw(img_pil)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            detected_count += 1

        gif_frames.append(np.array(img_pil))

    print(f"Bird detected in {detected_count}/{len(frame_paths)} frames")

    # Save GIF
    imageio.mimsave(output_path, gif_frames, fps=fps)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize VAE reconstructions and other visualizations")
    parser.add_argument("--vod_dir", type=str, required=True,
                        help="Path to the VOD directory containing frame data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the VAE checkpoint (required for visualize_reconstruction)")
    
    args = parser.parse_args()
    
    vod_dir = args.vod_dir
    checkpoint = args.checkpoint
    
    if checkpoint:
        visualize_reconstruction(checkpoint, vod_dir, num_frames=90)
    visualize_detection(vod_dir, num_frames=90)
    visualize_downsample(vod_dir, scale_factor=0.5, modes=("area", "bilinear"))
