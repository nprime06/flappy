"""Visualization utilities for autoencoder development."""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def load_random_frame(vod_dir):
    """Load a random frame from the dataset as a tensor in [-1, 1]."""
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


def visualize_downsample(vod_dir, scale_factor=0.5, modes=("area", "bilinear")):
    """Compare original frame with downsampled versions using different modes."""
    tensor, path = load_random_frame(vod_dir)
    print(f"Loaded: {path}")
    print(f"Original shape: {tensor.shape}")

    # Add batch dim for interpolate
    tensor_batch = tensor.unsqueeze(0)

    n_cols = 1 + len(modes)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Original
    axes[0].imshow(tensor_to_display(tensor))
    axes[0].set_title(f"Original\n{tensor.shape[1]}x{tensor.shape[2]}")
    axes[0].axis("off")

    # Downsampled versions
    for i, mode in enumerate(modes):
        downsampled = F.interpolate(tensor_batch, scale_factor=scale_factor, mode=mode)
        downsampled = downsampled.squeeze(0)
        axes[i + 1].imshow(tensor_to_display(downsampled))
        axes[i + 1].set_title(f"{mode}\n{downsampled.shape[1]}x{downsampled.shape[2]}")
        axes[i + 1].axis("off")
        print(f"{mode} shape: {downsampled.shape}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    vod_dir = "/Users/william/Desktop/Random/flappy/vod"
    visualize_downsample(vod_dir, scale_factor=0.5)
