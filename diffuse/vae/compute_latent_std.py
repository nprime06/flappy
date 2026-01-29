"""Compute latent std for a trained VAE checkpoint."""

import sys
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.vae import VAE


def main():
    checkpoint = "/Users/william/Desktop/Random/flappy/diffuse/vae/runs/vae_20260125_125114/checkpoints/latest.pt"
    vod_dir = "/Users/william/Desktop/Random/flappy/vod"
    num_samples = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint and extract model config
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]

    # Build model
    vae = VAE(
        image_channels=model_config["image_channels"],
        hidden_channels=model_config["hidden_channels"],
        latent_channels=model_config["latent_channels"],
        num_layers=model_config["num_layers"],
    ).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    # Gather random frame paths
    frame_paths = list(Path(vod_dir).glob("*/*/frames/*.png"))
    frame_paths = random.sample(frame_paths, min(num_samples, len(frame_paths)))
    print(f"Using {len(frame_paths)} frames")

    # Transform (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Encode all frames
    latents = []
    with torch.no_grad():
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            encoded = vae.encoder(x)
            z, _, _ = vae.reparameterize(encoded, sample=False)  # use mean
            latents.append(z)

    latents = torch.cat(latents, dim=0)
    std = latents.std().item()
    scaling_factor = 1.0 / std

    print(f"Latent shape: {latents.shape}")
    print(f"Latent mean: {latents.mean().item():.4f}")
    print(f"Latent std: {std:.4f}")
    print(f"Scaling factor (1/std): {scaling_factor:.6f}")


if __name__ == "__main__":
    main()
