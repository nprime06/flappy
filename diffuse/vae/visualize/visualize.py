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

# Constants
VISUALIZE_DIR = Path(__file__).parent

# Bird detection parameters
BIRD_X_MIN = 50
BIRD_X_MAX = 100
BIRD_HEIGHT = 35
BIRD_Y_OFFSET = 23
BIRD_COLOR_RANGE = {
    "r_min": 0.8,
    "g_min": 0.3,
    "g_max": 0.6,
    "b_max": 0.3,
}


# Helper functions
def tensor_to_display(tensor):
    """Convert tensor from [-1, 1] to numpy array for display [0, 1]."""
    img = (tensor + 1) / 2
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def load_frames_from_episode(vod_dir, num_frames):
    """Load contiguous frames from a random episode directory.
    
    Returns:
        tuple: (batch tensor, frame_paths, episode_dir, start_idx)
    """
    episode_dirs = sorted(Path(vod_dir).glob("*/*/frames"))
    episode_dir = random.choice(episode_dirs)
    frame_paths = sorted(episode_dir.glob("*.png"))
    
    max_start = max(0, len(frame_paths) - num_frames)
    start_idx = random.randint(0, max_start)
    frame_paths = frame_paths[start_idx:start_idx + num_frames]
    
    print(f"Episode: {episode_dir}")
    print(f"Frames {start_idx} to {start_idx + len(frame_paths) - 1}")
    
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        tensor = transforms.ToTensor()(img) * 2 - 1  # [-1, 1]
        frames.append(tensor)
    
    batch = torch.stack(frames)
    return batch, frame_paths, episode_dir, start_idx


def save_gif(frames, output_path, fps=30):
    """Save a list of numpy arrays as a GIF."""
    output_path = str(output_path)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved GIF to: {output_path}")


def load_vae_model(checkpoint_path, device="cpu"):
    """Load VAE model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    
    return model


def detect_bird_bbox(img_tensor):
    """Detect bird bounding box from image tensor.
    
    Args:
        img_tensor: Tensor in [-1, 1] format
        
    Returns:
        tuple or None: (x_min, y_min, x_max, y_max) or None if not detected
    """
    img = (img_tensor + 1) / 2  # [0, 1]
    r, g, b = img[0], img[1], img[2]
    
    bird_mask = (
        (r > BIRD_COLOR_RANGE["r_min"]) &
        (g > BIRD_COLOR_RANGE["g_min"]) &
        (g < BIRD_COLOR_RANGE["g_max"]) &
        (b < BIRD_COLOR_RANGE["b_max"])
    )
    
    if not bird_mask.any():
        return None
    
    H, W = bird_mask.shape
    rows = bird_mask.any(dim=1).nonzero(as_tuple=True)[0]
    y_min = rows.min().item() - BIRD_Y_OFFSET
    y_max = min(H - 1, y_min + BIRD_HEIGHT)
    
    return (BIRD_X_MIN, y_min, BIRD_X_MAX, y_max)


# Visualization functions
def visualize_downsample(vod_dir, scale_factor=0.5, modes=("area", "bilinear"),
                         output_path=None, num_frames=30, fps=30):
    """Compare original frames with downsampled versions using different modes and create a GIF."""
    if output_path is None:
        output_path = VISUALIZE_DIR / "downsample.gif"
    
    batch, frame_paths, episode_dir, start_idx = load_frames_from_episode(vod_dir, num_frames)
    print(f"Original shape: {batch.shape}")
    
    gif_frames = []
    for i in range(len(batch)):
        frame_tensor = batch[i]
        frame_batch = frame_tensor.unsqueeze(0)
        
        orig_display = tensor_to_display(frame_tensor)
        downsampled_displays = []
        
        for mode in modes:
            downsampled = F.interpolate(frame_batch, scale_factor=scale_factor, mode=mode).squeeze(0)
            downsampled_upsampled = F.interpolate(
                downsampled.unsqueeze(0),
                size=frame_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            downsampled_displays.append(tensor_to_display(downsampled_upsampled))
            
            if i == 0:
                print(f"{mode} shape: {downsampled.shape}")
        
        combined = np.concatenate([orig_display] + downsampled_displays, axis=1)
        gif_frames.append((combined * 255).astype(np.uint8))
    
    save_gif(gif_frames, output_path, fps)
    
    # Show first frame comparison
    n_cols = 1 + len(modes)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    axes[0].imshow(tensor_to_display(batch[0]))
    axes[0].set_title(f"Original\n{batch.shape[2]}x{batch.shape[3]}")
    axes[0].axis("off")
    
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
    
    model = load_vae_model(checkpoint_path, device)
    batch, frame_paths, episode_dir, start_idx = load_frames_from_episode(vod_dir, num_frames)
    batch = batch.to(device)
    
    with torch.no_grad():
        encoded = model.encoder(batch)
        z, mean, logvar = model.reparameterize(encoded, sample=False)
        recon = model.decoder(z).cpu()
    
    if recon.shape[-2:] != batch.shape[-2:]:
        recon = F.interpolate(recon, size=batch.shape[-2:], mode='bilinear', align_corners=False)
    
    batch = batch.cpu()
    l1 = F.l1_loss(recon, batch).item()
    print(f"Average L1: {l1:.4f}")
    
    gif_frames = []
    for i in range(len(batch)):
        orig = tensor_to_display(batch[i])
        rec = tensor_to_display(recon[i])
        combined = np.concatenate([orig, rec], axis=1)
        gif_frames.append((combined * 255).astype(np.uint8))
    
    save_gif(gif_frames, output_path, fps)
    
    # Show first frame comparison
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
    
    batch, frame_paths, episode_dir, start_idx = load_frames_from_episode(vod_dir, num_frames)
    transform = transforms.ToTensor()
    
    gif_frames = []
    detected_count = 0
    
    for path in frame_paths:
        img_pil = Image.open(path).convert("RGB")
        img_tensor = transform(img_pil) * 2 - 1
        
        bbox = detect_bird_bbox(img_tensor)
        
        draw = ImageDraw.Draw(img_pil)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            detected_count += 1
        
        gif_frames.append(np.array(img_pil))
    
    print(f"Bird detected in {detected_count}/{len(frame_paths)} frames")
    save_gif(gif_frames, output_path, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize VAE reconstructions and other visualizations")
    parser.add_argument("--vod_dir", type=str, required=True,
                        help="Path to the VOD directory containing frame data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the VAE checkpoint (required for visualize_reconstruction)")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        visualize_reconstruction(args.checkpoint, args.vod_dir, num_frames=90)
    visualize_detection(args.vod_dir, num_frames=90)
    visualize_downsample(args.vod_dir, scale_factor=0.5, modes=("area", "bilinear"))
