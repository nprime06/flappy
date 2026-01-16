"""Flow matching training for world model (next frame prediction)."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.ae import VAE
from nn.resunet import ResUNet
from ngen.ngen_data import TraceDataset
from ngen.loss import flow_matching_loss
from ngen.sampler import ReflowPairGenerator

model_config = {
    "in_channels": 4,           # latent_channels from VAE
    "hidden_channels": 64,
    "num_layers": 2,            # limited by odd latent width (18)
    "embed_dim": 128,
    "num_classes": 2,           # flappy bird: 0=no-flap, 1=flap
    "context_frames": 4,        # k past frames
    "num_aug_bins": 16,
}

train_config = {
    "lr": 1e-4,
    "num_epochs": 100,
    "batch_size": 32,
    "max_aug_std": 0.5,
    "latent_mean": 10.1880,
    "latent_std": 13.3726,
    "log_interval": 1,
    "checkpoint_interval": 10,
    "num_workers": 4,
    "reflow_steps": 50,         # Euler steps for reflow pair generation
    "vae_checkpoint": "/home/willzhao/flappy/diffuse/ae/runs/vae_20260115_022006/checkpoints/latest.pt",
    "data_dir": "/home/willzhao/flappy/vod",
    "runs_dir": "/home/willzhao/flappy/diffuse/ngen/runs",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


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
    """Load a pre-trained flow model (for reflow)."""
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
    return model


def vae_encode(vae, x, latent_mean, latent_std):
    """Encode image to normalized latent mean (deterministic)."""
    encoded = vae.encoder(x)
    z, _, _ = vae.reparameterize(encoded, sample=False)
    z = (z - latent_mean) / latent_std
    return z


def train(run_dir=None, reflow_checkpoint=None):
    """
    Train flow matching model.

    Args:
        run_dir: Directory to save checkpoints (None = create new)
        reflow_checkpoint: Path to pre-trained model for reflow training.
                          If None, use standard flow matching (z_0 ~ N(0,1)).
    """
    device = train_config["device"]

    # Load frozen VAE
    print(f"Loading VAE from {train_config['vae_checkpoint']}")
    vae = load_vae(train_config["vae_checkpoint"], device)

    # Load reflow model if specified
    reflow_generator = None
    if reflow_checkpoint is not None:
        print(f"Loading reflow model from {reflow_checkpoint}")
        reflow_model = load_flow_model(reflow_checkpoint, device)
        reflow_generator = ReflowPairGenerator(reflow_model, num_steps=train_config["reflow_steps"])
        print("Reflow mode enabled: using pre-trained model to generate (z_0, z_1) pairs")

    # Build flow model
    model = ResUNet(
        in_channels=model_config["in_channels"],
        hidden_channels=model_config["hidden_channels"],
        num_layers=model_config["num_layers"],
        embed_dim=model_config["embed_dim"],
        num_classes=model_config["num_classes"],
        context_channels=model_config["context_frames"] * model_config["in_channels"],
        num_aug_bins=model_config["num_aug_bins"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Flow model has {num_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=train_config["lr"])

    # Setup run directory
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "reflow" if reflow_checkpoint else "ngen"
        run_dir = os.path.join(train_config["runs_dir"], f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "training_log.jsonl")
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    config_path = os.path.join(run_dir, "config.json")

    print(f"Run directory: {run_dir}")

    # Save config
    if not os.path.exists(config_path):
        config = {
            "model_config": model_config,
            "train_config": train_config,
            "reflow_checkpoint": reflow_checkpoint,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Try to resume from checkpoint
    start_epoch = 0
    total_wall_time = 0.0

    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        total_wall_time = ckpt.get("wall_time_s", 0.0)
        print(f"Resumed at epoch {start_epoch}")

    # Setup data
    dataset = TraceDataset(train_config["data_dir"], k=model_config["context_frames"])
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")

    log_file = open(log_path, "a", buffering=1)
    train_start_t = time.perf_counter()

    latent_mean = train_config["latent_mean"]
    latent_std = train_config["latent_std"]
    max_aug_std = train_config["max_aug_std"]
    num_aug_bins = model_config["num_aug_bins"]

    model.train()
    for epoch in tqdm(range(start_epoch, train_config["num_epochs"])):
        epoch_loss = 0.0
        epoch_v_pred_norm = 0.0
        epoch_v_target_norm = 0.0

        for past_frames, current_frame, action in dataloader:
            past_frames = past_frames.to(device)
            current_frame = current_frame.to(device)
            action = action.to(device)

            B, k = past_frames.shape[:2]

            # Encode through frozen VAE
            with torch.no_grad():
                z_target = vae_encode(vae, current_frame, latent_mean, latent_std)
                past_flat = past_frames.flatten(0, 1)
                z_cond = vae_encode(vae, past_flat, latent_mean, latent_std)
                z_cond = z_cond.unflatten(0, (B, k)).flatten(1, 2)

            # Noise augmentation on conditioning
            aug_level = torch.randint(0, num_aug_bins, (B,), device=device)
            aug_std = aug_level.float() / num_aug_bins * max_aug_std
            z_cond = z_cond + torch.randn_like(z_cond) * aug_std.view(B, 1, 1, 1)

            # Get z_0 (either from N(0,1) or from reflow generator)
            if reflow_generator is not None:
                # Reflow: generate (z_0, z_1) pairs using pre-trained model
                z_0, z_1 = reflow_generator.generate(z_target.shape, z_cond, action, aug_level)
                # Use generated z_1 as target instead of real data
                z_target = z_1
            else:
                z_0 = None  # Will sample from N(0,1) in loss function

            # Compute loss
            loss, info = flow_matching_loss(model, z_target, z_cond, action, aug_level, z_0=z_0)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_v_pred_norm += info["v_pred_norm"]
            epoch_v_target_norm += info["v_target_norm"]

        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        avg_v_pred_norm = epoch_v_pred_norm / n_batches
        avg_v_target_norm = epoch_v_target_norm / n_batches
        wall_time_s = total_wall_time + (time.perf_counter() - train_start_t)

        # Log every epoch
        log_file.write(json.dumps({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "v_pred_norm": avg_v_pred_norm,
            "v_target_norm": avg_v_target_norm,
            "wall_time_s": wall_time_s,
        }) + "\n")

        if (epoch + 1) % train_config["log_interval"] == 0:
            print(f"Epoch {epoch+1:4d}/{train_config['num_epochs']} | Loss: {avg_loss:.6f}")

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
    parser.add_argument("--reflow", type=str, default=None,
                        help="Path to pre-trained model checkpoint for reflow training")
    args = parser.parse_args()
    train(run_dir=args.run_dir, reflow_checkpoint=args.reflow)
