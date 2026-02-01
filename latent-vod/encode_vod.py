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
from diffuse.nn.vae import VAE


def load_latent_statistics(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    latent_mean = ckpt.get("latent_mean")
    latent_std = ckpt.get("latent_std")
    return latent_mean, latent_std

def load_vae(checkpoint_path, device):
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


def compute_dataset_statistics(output_dir, sample_size):
    stats = {
        "total_steps": 0,
        "action_0_count": 0,
        "action_1_count": 0,
        "done_0_count": 0,
        "done_1_count": 0,
    }
    categories_sampled = []

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

                    action = rec.get("action", 0)
                    if action == 0:
                        stats["action_0_count"] += 1
                    else:
                        stats["action_1_count"] += 1

                    terminated = rec.get("terminated", False)
                    if terminated:
                        stats["done_1_count"] += 1
                    else:
                        stats["done_0_count"] += 1

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
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    vae = load_vae(args.vae_checkpoint, device)
    latent_mean, latent_std = load_latent_statistics(args.vae_checkpoint)

    output_dir = Path(args.output_dir) # wipe output dir
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
    print(f"saved encoding config to {config_path}")

    vod_dir = Path(args.vod_dir)
    runs = list(vod_dir.glob("*/*/"))
    runs = [r for r in runs if (r / "frames").exists() and (r / "run_info.jsonl").exists()]

    print(f"found {len(runs)} runs to encode")

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

    print(f"\nencoding complete! output directory: {output_dir}")

    stats_result = compute_dataset_statistics(output_dir, sample_size=128)

    config.update(stats_result)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved dataset statistics to {config_path}")
    print(f"action weight: {stats_result['action_weight']}")
    print(f"done pos weight: {stats_result['done_pos_weight']}")


if __name__ == "__main__":
    main()
