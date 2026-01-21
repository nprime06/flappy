"""Flow matching training for world model (next frame prediction)."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.ae import VAE
from nn.resunet import ResUNet
from ngen.ngen_data import TraceDataset
from ngen.loss import flow_matching_loss
from ngen.sampler import ReflowPairGenerator


def setup_ddp():
    """Initialize DDP. Returns (rank, local_rank, world_size, device)."""
    if "RANK" in os.environ:
        # Launched via torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, local_rank, world_size, device
    else:
        # Single GPU fallback (direct python invocation)
        return 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0)."""
    return rank == 0


model_config = {
    "in_channels": 4,           # latent_channels from VAE
    "hidden_channels": 64,
    "num_layers": 2,            # limited by odd latent width (18)
    "embed_dim": 128,
    "num_classes": 2,           # flappy bird: 0=no-flap, 1=flap
    "context_frames": 8,        # k past frames (increased from 2)
    "num_aug_bins": 16,
    "cfg_dropout_prob": 0.1,    # CFG: zero out conditioning 10% of time
    "use_done_head": True,      # Enable termination detection head
    "done_loss_weight": 0.1,    # Weight for done head BCE loss
}

train_config = {
    "lr": 1e-4,
    "num_epochs": 40,
    "batch_size": 256,
    "max_aug_std": 0.5,
    "latent_mean": 10.1880,
    "latent_std": 13.3726,
    "log_interval": 1,
    "checkpoint_interval": 5,
    "num_workers": 2,
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
    Train flow matching model with optional DDP support.

    Args:
        run_dir: Directory to save checkpoints (None = create new)
        reflow_checkpoint: Path to pre-trained model for reflow training.
                          If None, use standard flow matching (z_0 ~ N(0,1)).
    """
    # Setup DDP (works for single GPU too)
    rank, local_rank, world_size, device = setup_ddp()
    is_main = is_main_process(rank)

    # Load frozen VAE (not wrapped in DDP - inference only)
    if is_main:
        print(f"Loading VAE from {train_config['vae_checkpoint']}")
    vae = load_vae(train_config["vae_checkpoint"], device)
    # vae.encoder = torch.compile(vae.encoder)  # Disabled: causes OOM on multi-GPU DDP

    # Load reflow model if specified (not wrapped in DDP - inference only)
    reflow_generator = None
    if reflow_checkpoint is not None:
        if is_main:
            print(f"Loading reflow model from {reflow_checkpoint}")
        reflow_model = load_flow_model(reflow_checkpoint, device)
        reflow_generator = ReflowPairGenerator(reflow_model, num_steps=train_config["reflow_steps"])
        if is_main:
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
        use_done_head=model_config.get("use_done_head", False),
    ).to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Flow model has {num_params:,} parameters")

    # Wrap in DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Access raw model for state_dict (unwrap DDP if needed)
    raw_model = model.module if world_size > 1 else model

    optimizer = AdamW(raw_model.parameters(), lr=train_config["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"], eta_min=1e-6)

    # Setup run directory (rank 0 only creates directories)
    if is_main:
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "reflow" if reflow_checkpoint else "ngen"
            run_dir = os.path.join(train_config["runs_dir"], f"{prefix}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Broadcast run_dir to all ranks (needed for checkpoint paths)
    if world_size > 1:
        run_dir_list = [run_dir] if is_main else [None]
        dist.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_path = os.path.join(run_dir, "training_log.jsonl")
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    config_path = os.path.join(run_dir, "config.json")

    if is_main:
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

    # Try to resume from checkpoint (all ranks load, but print on main only)
    start_epoch = 0
    total_wall_time = 0.0

    if os.path.exists(latest_ckpt_path):
        if is_main:
            print(f"Resuming from {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])  # Load to raw model, not DDP wrapper
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        total_wall_time = ckpt.get("wall_time_s", 0.0)
        if is_main:
            print(f"Resumed at epoch {start_epoch}")

    # Setup data with DistributedSampler for multi-GPU
    include_done = model_config.get("use_done_head", False)
    dataset = TraceDataset(train_config["data_dir"], k=model_config["context_frames"], include_done=include_done)

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=train_config["batch_size"],  # Per-GPU batch size
            sampler=sampler,
            num_workers=train_config["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=train_config["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )

    if is_main:
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Batches per epoch: {len(dataloader)}")

    # Only main process opens log file
    log_file = open(log_path, "a", buffering=1) if is_main else None
    train_start_t = time.perf_counter()

    latent_mean = train_config["latent_mean"]
    latent_std = train_config["latent_std"]
    max_aug_std = train_config["max_aug_std"]
    num_aug_bins = model_config["num_aug_bins"]

    cfg_dropout_prob = model_config.get("cfg_dropout_prob", 0.0)
    use_done_head = model_config.get("use_done_head", False)
    done_loss_weight = model_config.get("done_loss_weight", 0.1)

    model.train()
    epoch_iterator = tqdm(range(start_epoch, train_config["num_epochs"])) if is_main else range(start_epoch, train_config["num_epochs"])
    for epoch in epoch_iterator:
        # CRITICAL: Set epoch for proper shuffling in DistributedSampler
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_v_pred_norm = 0.0
        epoch_v_target_norm = 0.0
        epoch_done_loss = 0.0

        for batch in dataloader:
            # Unpack batch (with or without done labels)
            if use_done_head:
                past_frames, current_frame, action, done_labels = batch
                done_labels = done_labels.to(device)
            else:
                past_frames, current_frame, action = batch
                done_labels = None

            past_frames = past_frames.to(device)
            current_frame = current_frame.to(device)
            action = action.to(device)

            B, k = past_frames.shape[:2]

            # Encode through frozen VAE (with bfloat16 for speed)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_target = vae_encode(vae, current_frame, latent_mean, latent_std)
                past_flat = past_frames.flatten(0, 1)
                z_cond = vae_encode(vae, past_flat, latent_mean, latent_std)
                z_cond = z_cond.unflatten(0, (B, k)).flatten(1, 2)

            # Noise augmentation on conditioning
            aug_level = torch.randint(0, num_aug_bins, (B,), device=device)
            aug_std = aug_level.float() / num_aug_bins * max_aug_std
            z_cond = z_cond + torch.randn_like(z_cond) * aug_std.view(B, 1, 1, 1)

            # CFG dropout: zero out conditioning for a fraction of samples
            if cfg_dropout_prob > 0:
                cfg_mask = torch.rand(B, device=device) < cfg_dropout_prob
                z_cond = torch.where(cfg_mask.view(B, 1, 1, 1), torch.zeros_like(z_cond), z_cond)

            # Get z_0 (either from N(0,1) or from reflow generator)
            if reflow_generator is not None:
                # Reflow: infer z_0 from real z_target by flowing backward
                z_0 = reflow_generator.generate(z_target, z_cond, action, aug_level)
                # z_target stays as real data
            else:
                z_0 = None  # Will sample from N(0,1) in loss function

            # Compute loss (with bfloat16 for speed)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, info = flow_matching_loss(
                    model, z_target, z_cond, action, aug_level, z_0=z_0,
                    done_labels=done_labels, done_loss_weight=done_loss_weight
                )

            # Backward (autocast handles gradient dtype automatically)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            epoch_loss += loss.item()
            epoch_v_pred_norm += info["v_pred_norm"]
            epoch_v_target_norm += info["v_target_norm"]
            if "done_loss" in info:
                epoch_done_loss += info["done_loss"]

        # Step the LR scheduler
        scheduler.step()

        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        avg_v_pred_norm = epoch_v_pred_norm / n_batches
        avg_v_target_norm = epoch_v_target_norm / n_batches
        avg_done_loss = epoch_done_loss / n_batches if use_done_head else 0.0
        wall_time_s = total_wall_time + (time.perf_counter() - train_start_t)
        current_lr = scheduler.get_last_lr()[0]

        # Log every epoch (main process only)
        if is_main:
            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "v_pred_norm": avg_v_pred_norm,
                "v_target_norm": avg_v_target_norm,
                "lr": current_lr,
                "wall_time_s": wall_time_s,
            }
            if use_done_head:
                log_entry["done_loss"] = avg_done_loss
            log_file.write(json.dumps(log_entry) + "\n")

            if (epoch + 1) % train_config["log_interval"] == 0:
                print(f"Epoch {epoch+1:4d}/{train_config['num_epochs']} | Loss: {avg_loss:.6f}")

        # Checkpoint saving (main process only, with barrier for sync)
        if (epoch + 1) % train_config["checkpoint_interval"] == 0:
            # Ensure all ranks are synced before saving
            if world_size > 1:
                dist.barrier()

            if is_main:
                ckpt = {
                    "model": raw_model.state_dict(),  # Save raw model, not DDP wrapper
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "wall_time_s": wall_time_s,
                    "model_config": model_config,
                    "train_config": train_config,
                }
                torch.save(ckpt, latest_ckpt_path)
                torch.save(ckpt, os.path.join(checkpoint_dir, f"ep_{epoch+1:05d}.pt"))

    # Cleanup
    if is_main:
        log_file.close()
        print(f"\nTraining complete. Run directory: {run_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory to resume from (creates new if not specified)")
    parser.add_argument("--reflow", type=str, default=None,
                        help="Path to pre-trained model checkpoint for reflow training")
    args = parser.parse_args()
    train(run_dir=args.run_dir, reflow_checkpoint=args.reflow)
