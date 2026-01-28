import argparse
import json
import os
import random
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

# Fallback values (for backward compatibility)
DEFAULT_LATENT_MEAN = 0.4735
DEFAULT_LATENT_STD = 1.5931

def load_latent_statistics(checkpoint_path):
    """Load latent statistics from run directory's config.json.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        tuple of (latent_mean, latent_std)
    """
    checkpoint_path = Path(checkpoint_path)
    # Checkpoint is in runs/vae_*/checkpoints/latest.pt
    # Config is in runs/vae_*/config.json
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / "config.json"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            latent_mean = config.get("latent_mean", DEFAULT_LATENT_MEAN)
            latent_std = config.get("latent_std", DEFAULT_LATENT_STD)
            print(f"Loaded latent statistics from {config_path}: mean={latent_mean:.4f}, std={latent_std:.4f}")
            return latent_mean, latent_std
    else:
        print(f"Warning: {config_path} not found, using default statistics")
        return DEFAULT_LATENT_MEAN, DEFAULT_LATENT_STD

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

def encode(vae, x, latent_mean, latent_std):
    encoded = vae.encoder(x)
    z, _, _ = vae.reparameterize(encoded, sample=False)
    z = (z - latent_mean) / latent_std
    return z


def encode_run(vae, frames_dir, latent_mean, latent_std, batch_size, device):
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
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents = encode(vae, batch_tensor, latent_mean, latent_std)

        all_latents.append(latents.cpu())

    return torch.cat(all_latents, dim=0)


def compute_dataset_statistics(output_dir, sample_size=100):
    """Sample run files and compute action/done distribution statistics.

    Args:
        output_dir: Path to latent-vod output directory
        sample_size: Number of runs to sample per category directory

    Returns:
        dict with action_weight, done_pos_weight, and raw statistics
    """
    stats = {
        "total_steps": 0,
        "action_0_count": 0,
        "action_1_count": 0,
        "done_0_count": 0,
        "done_1_count": 0,
    }
    categories_sampled = []

    # Find all p_stim_*_p_freeze_* directories
    for category_dir in output_dir.glob("p_stim_*_p_freeze_*/"):
        runs = list(category_dir.glob("*/run_info.jsonl"))
        if not runs:
            continue

        sampled = random.sample(runs, min(sample_size, len(runs)))
        categories_sampled.append(category_dir.name)

        for run_file in sampled:
            with open(run_file, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    if "step" not in rec:
                        continue  # Skip reset event

                    stats["total_steps"] += 1

                    # Count actions
                    action = rec.get("action", 0)
                    if action == 0:
                        stats["action_0_count"] += 1
                    else:
                        stats["action_1_count"] += 1

                    # Count done states (use terminated flag)
                    terminated = rec.get("terminated", False)
                    if terminated:
                        stats["done_1_count"] += 1
                    else:
                        stats["done_0_count"] += 1

    # Compute weights (inverse ratios)
    action_weight = stats["action_0_count"] / max(stats["action_1_count"], 1)
    done_pos_weight = stats["done_0_count"] / max(stats["done_1_count"], 1)

    return {
        "action_weight": round(action_weight, 2),
        "done_pos_weight": round(done_pos_weight, 2),
        "statistics": {
            **stats,
            "sample_size": sample_size,
            "categories_sampled": categories_sampled,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading VAE from {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device)
    
    # Load latent statistics from config.json
    latent_mean, latent_std = load_latent_statistics(args.vae_checkpoint)

    # first erase all latents in latent-vod
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        for item in output_dir.iterdir(): 
            if item.is_file() and (item.suffix == '.py' or item.suffix == '.sh'):
                continue
            if item.is_dir() and item.name == 'encode-logs':
                continue
            if item.is_dir():
                shutil.rmtree(item) # remove frame directories
            else:
                item.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "encode_config.json"
    config = {
        "vae_checkpoint": args.vae_checkpoint,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
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
        latents = encode_run(vae, frames_dir, latent_mean, latent_std,
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

    # Compute dataset statistics for training weights
    print("\nComputing dataset statistics...")
    stats_result = compute_dataset_statistics(output_dir, sample_size=100)

    # Update config with statistics
    config.update(stats_result)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Dataset statistics:")
    print(f"  Total steps sampled: {stats_result['statistics']['total_steps']}")
    print(f"  Action distribution: {stats_result['statistics']['action_0_count']} (0) / {stats_result['statistics']['action_1_count']} (1)")
    print(f"  Done distribution: {stats_result['statistics']['done_0_count']} (0) / {stats_result['statistics']['done_1_count']} (1)")
    print(f"  Computed action_weight: {stats_result['action_weight']}")
    print(f"  Computed done_pos_weight: {stats_result['done_pos_weight']}")
    print(f"Updated {config_path}")


if __name__ == "__main__":
    main()
