"""VAE training script for video frames."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.ae import VAE
from ae.ae_data import FramesDataset

model_config = {
    "image_channels": 3,
    "hidden_channels": 16,
    "latent_channels": 4,
    "num_layers": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

train_config = {
    "lr": 1e-4,
    "num_epochs": 200,
    "batch_size": 256,
    "kl_weight": 1e-3,
    "grad_weight": 1.0,
    "bird_weight": 10.0,  # extra weight for bird pixels
    "log_interval": 1,
    "checkpoint_interval": 10,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # "data_dir": "/Users/william/Desktop/Random/flappy/vod",
    # "runs_dir": "/Users/william/Desktop/Random/flappy/diffuse/ae/runs",
    "data_dir": "/home/willzhao/flappy/vod",
    "runs_dir": "/home/willzhao/flappy/diffuse/ae/runs",
}


def gradient_loss(recon, target):
    """Compute L1 loss on spatial gradients (edges)."""
    # Horizontal gradients
    recon_dx = recon[:, :, :, 1:] - recon[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    # Vertical gradients
    recon_dy = recon[:, :, 1:, :] - recon[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy)


def get_bird_weight_mask(target, bird_weight=10.0, bird_x_min=50, bird_x_max=100, bird_height=35):
    """Create weight mask with higher weight in bird bounding box.

    The bird has a fixed x position (world scrolls past it), so we only
    detect the top edge (y_min) and use fixed x bounds and height.

    Args:
        target: tensor in [-1, 1], shape (B, 3, H, W)
        bird_weight: weight multiplier for bird region
        bird_x_min, bird_x_max: fixed horizontal bounds
        bird_height: fixed height of bounding box

    Returns:
        weight mask shape (B, 1, H, W)
    """
    # Convert to [0, 1] for color detection
    img = (target + 1) / 2
    r, g, b = img[:, 0], img[:, 1], img[:, 2]

    # Detect orange bird body
    bird_mask = (r > 0.8) & (g > 0.3) & (g < 0.6) & (b < 0.3)

    B, _, H, W = target.shape
    weights = torch.ones(B, 1, H, W, device=target.device)

    for i in range(B):
        mask_i = bird_mask[i]
        if mask_i.any():
            # Find top edge of bird (offset by 23 to capture white head above orange)
            rows = mask_i.any(dim=1).nonzero(as_tuple=True)[0]
            y_min = max(0, rows.min().item() - 23)
            y_max = min(H - 1, y_min + bird_height)

            weights[i, 0, y_min:y_max+1, bird_x_min:bird_x_max+1] = bird_weight

    return weights


def weighted_l1_loss(recon, target, weights):
    """L1 loss with per-pixel weights."""
    return (weights * (recon - target).abs()).mean()


def train(run_dir=None):
    model = VAE(
        image_channels=model_config["image_channels"],
        hidden_channels=model_config["hidden_channels"],
        latent_channels=model_config["latent_channels"],
        num_layers=model_config["num_layers"],
    ).to(model_config["device"])

    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Decoder has {decoder_params:,} parameters")
    print(f"Encoder has {encoder_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=train_config["lr"])

    # Setup run directory
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(train_config["runs_dir"], f"vae_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "training_log.jsonl")
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    config_path = os.path.join(run_dir, "config.json")

    print(f"Run directory: {run_dir}")

    # Save config
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(train_config, f, indent=2)

    # Try to resume from checkpoint
    start_epoch = 0
    total_wall_time = 0.0

    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=model_config["device"], weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        total_wall_time = ckpt.get("wall_time_s", 0.0)
        print(f"Resumed at epoch {start_epoch}")

    # Setup data
    dataset = FramesDataset(train_config["data_dir"])
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Dataset size: {len(dataset)} frames")
    print(f"Batches per epoch: {len(dataloader)}")

    log_file = open(log_path, "a", buffering=1)
    train_start_t = time.perf_counter()

    model.train()
    for epoch in tqdm(range(start_epoch, train_config["num_epochs"])):
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_grad = 0.0
        epoch_kl = 0.0

        for batch in dataloader:
            batch = batch.to(model_config["device"])

            encoded = model.encoder(batch)
            z, mean, logvar = model.reparameterize(encoded, sample=True)
            recon = model.decoder(z)

            # Downsample target if decoder outputs lower resolution
            target = batch
            if recon.shape[-2:] != batch.shape[-2:]:
                target = F.interpolate(batch, size=recon.shape[-2:], mode='area')

            # Weighted L1 loss: higher weight on bird region
            bird_weights = get_bird_weight_mask(target, bird_weight=train_config["bird_weight"])
            l1_loss = weighted_l1_loss(recon, target, bird_weights)
            grad_loss = gradient_loss(recon, target)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss = l1_loss + train_config["grad_weight"] * grad_loss + train_config["kl_weight"] * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_l1 += l1_loss.item()
            epoch_grad += grad_loss.item()
            epoch_kl += kl_loss.item()

        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_grad = epoch_grad / n_batches
        avg_kl = epoch_kl / n_batches
        wall_time_s = total_wall_time + (time.perf_counter() - train_start_t)

        # Log every epoch
        log_file.write(json.dumps({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "l1_loss": avg_l1,
            "grad_loss": avg_grad,
            "kl_loss": avg_kl,
            "wall_time_s": wall_time_s,
        }) + "\n")

        if (epoch + 1) % train_config["log_interval"] == 0:
            print(
                f"Epoch {epoch+1:4d}/{train_config['num_epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"L1: {avg_l1:.4f} | "
                f"Grad: {avg_grad:.4f} | "
                f"KL: {avg_kl:.4f}"
            )

        if (epoch + 1) % train_config["checkpoint_interval"] == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "wall_time_s": wall_time_s,
                "model_config": model_config,
                "train_config": train_config,
            }
            torch.save(ckpt, latest_ckpt_path)
            torch.save(ckpt, os.path.join(checkpoint_dir, f"ep_{epoch+1:05d}.pt"))

    log_file.close()
    print(f"\nTraining complete. Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory to resume from (creates new if not specified)")
    args = parser.parse_args()
    train(run_dir=args.run_dir)