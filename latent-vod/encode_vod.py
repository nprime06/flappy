"""Pre-compute and save VAE latents for efficient training.

This script encodes all frames through the VAE encoder once, saving normalized
latents to disk. Since encoding is deterministic (sample=False), this avoids
redundant computation during training.

Usage:
    python encode_vod.py --vod-dir /path/to/vod --output-dir /path/to/latent-vod
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffuse.nn.ae import VAE

LATENT_MEAN = 0.4755
LATENT_STD = 1.5959

def load_vae(checkpoint_path, device):
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

def encode(vae, x):
    encoded = vae.encoder(x)
    z, _, _ = vae.reparameterize(encoded, sample=False)
    z = (z - LATENT_MEAN) / LATENT_STD
    return z


def encode_run(vae, frames_dir, batch_size, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    frame_files = sorted(frames_dir.glob("*.png"))
    n_frames = len(frame_files)

    if n_frames == 0:
        return None

    all_latents = []
    for i in range(0, n_frames, batch_size):
        batch_files = frame_files[i:i + batch_size]

        batch_images = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            batch_images.append(transform(img))

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents = encode(vae, batch_tensor)

        all_latents.append(latents.cpu())

    return torch.cat(all_latents, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents for VOD data")
    parser.add_argument("--vod-dir", type=str, required=True,
                        help="Directory containing VOD runs")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save latent-encoded runs")
    parser.add_argument("--vae-checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of frames to encode per batch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VAE
    print(f"Loading VAE from {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device)

    # first erase all latents in latent-vod
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        for item in output_dir.iterdir(): 
            # Preserve script files (.py, .sh) and encode-logs directory
            if item.is_file() and (item.suffix == '.py' or item.suffix == '.sh'):
                continue
            if item.is_dir() and item.name == 'encode-logs':
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "encode_config.json"
    config = {
        "vae_checkpoint": args.vae_checkpoint,
        "latent_mean": LATENT_MEAN,
        "latent_std": LATENT_STD,
        "batch_size": args.batch_size,
        "encoded_at": datetime.now().isoformat(),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved encoding config to {config_path}")

    vod_dir = Path(args.vod_dir)
    runs = list(vod_dir.glob("*/*/"))
    runs = [r for r in runs if (r / "frames").exists() and (r / "run_info.jsonl").exists()]

    print(f"Found {len(runs)} runs to encode")

    encoded_count = 0

    for run_dir in tqdm(runs, desc="Encoding runs"):
        rel_path = run_dir.relative_to(vod_dir)
        out_run_dir = output_dir / rel_path

        frames_dir = run_dir / "frames"
        latents = encode_run(vae, frames_dir,
                            args.batch_size, device)

        out_run_dir.mkdir(parents=True, exist_ok=True)
        latents_path = out_run_dir / "latents.pt"
        torch.save(latents, latents_path)

        src_info = run_dir / "run_info.jsonl"
        dst_info = out_run_dir / "run_info.jsonl"
        shutil.copy(src_info, dst_info)

        encoded_count += 1

    print(f"\nEncoding complete!")
    print(f"  Encoded: {encoded_count} runs")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    main()
