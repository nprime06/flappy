"""Visualize bird bounding box detection on video frames."""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import imageio


def detect_bird_bbox(img_tensor, bird_x_min=50, bird_x_max=100, bird_height=35):
    """Detect bird bounding box from a single frame tensor in [-1, 1].

    The bird has a fixed x position (world scrolls past it), so we only
    detect the top edge (y_min) and use fixed x bounds and height.

    Returns (x_min, y_min, x_max, y_max) or None if no bird detected.
    """
    img = (img_tensor + 1) / 2  # [0, 1]
    r, g, b = img[0], img[1], img[2]

    # Detect orange bird body
    bird_mask = (r > 0.8) & (g > 0.3) & (g < 0.6) & (b < 0.3)

    if not bird_mask.any():
        return None

    H, W = bird_mask.shape

    # Find top edge of bird (y_min)
    rows = bird_mask.any(dim=1).nonzero(as_tuple=True)[0]
    y_min = rows.min().item()-23

    # Fixed x bounds and height
    y_max = min(H - 1, y_min + bird_height)

    return (bird_x_min, y_min, bird_x_max, y_max)


def visualize_detection(vod_dir, output_path="bird_detection.gif", num_frames=60, fps=30):
    """Create a GIF showing bird bounding box detection on video frames."""

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

    for path in frame_paths:
        # Load and convert to tensor
        img_pil = Image.open(path).convert("RGB")
        img_tensor = transform(img_pil) * 2 - 1  # [-1, 1]

        # Detect bird bbox
        bbox = detect_bird_bbox(img_tensor)

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
    vod_dir = "/Users/william/Desktop/Random/flappy/vod"
    visualize_detection(vod_dir, output_path="bird_detection.gif", num_frames=120)
