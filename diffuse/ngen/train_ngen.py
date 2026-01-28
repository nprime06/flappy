"""Flow matching training for world model (next frame prediction)."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.resunet import ResUNet
from ngen.ngen_data import LatentTraceDataset
from ngen.loss import flow_matching_loss
from ngen.sampler import ReflowPairGenerator

# work directly with normalized latents

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
    "hidden_channels": 128,
    "num_layers": 2,            # limited by odd latent width (18)
    "embed_dim": 128,
    "act_embed_dim": 16,
    "num_classes": 2,           # flappy bird: 0=no-flap, 1=flap
    "context_size": 8,               # k frames and actions
    "num_aug_bins": 16,
}

train_config = {
    "lr": 1e-4,
    "num_epochs": 1000,
    "batch_size": 512,
    "max_aug_std": 0.5,
    "checkpoint_interval": 200,
    "num_workers": 4,

    "reflow_steps": 50,
    "cfg_dropout_prob": 0.1,
    "done_loss_weight": 0.1, # Weight for done head BCE loss
    "done_pos_weight": 30.0, # BCE pos_weight for done=1 (~avg episode length)
    "action_weight": 17.0, # extra weight for action=1 to fix class imbalance
    "runs_dir": "/home/willzhao/flappy/diffuse/ngen/runs",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def load_weights_from_encode_config(latent_vod_dir):
    """Load action_weight and done_pos_weight from encode_config.json if available.

    Args:
        latent_vod_dir: Path to latent-vod directory

    Returns:
        dict with action_weight and done_pos_weight (may be None if not found)
    """
    from pathlib import Path
    config_path = Path(latent_vod_dir) / "encode_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        return {
            "action_weight": config.get("action_weight"),
            "done_pos_weight": config.get("done_pos_weight"),
        }
    return {}


def load_flow_model(checkpoint_path, device):
    """Load a pre-trained flow model (for reflow)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = ResUNet(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        embed_dim=cfg["embed_dim"],
        act_embed_dim=cfg["act_embed_dim"],
        num_classes=cfg["num_classes"],
        context_size=cfg["context_size"],
        num_aug_bins=cfg["num_aug_bins"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def train(run_dir=None, reflow_checkpoint=None, latent_vod=None):
    rank, local_rank, world_size, device = setup_ddp()
    is_main = is_main_process(rank)

    # load reflow model (no ddp, inference only)
    reflow_generator = None
    if reflow_checkpoint is not None:
        if is_main:
            print(f"Loading reflow model from {reflow_checkpoint}")
        reflow_model = load_flow_model(reflow_checkpoint, device)
        reflow_generator = ReflowPairGenerator(reflow_model, num_steps=train_config["reflow_steps"])
        if is_main:
            print("Reflow mode enabled: using pre-trained model to generate (z_0, z_1) pairs")

    model = ResUNet(
        in_channels=model_config["in_channels"],
        hidden_channels=model_config["hidden_channels"],
        num_layers=model_config["num_layers"],
        embed_dim=model_config["embed_dim"],
        act_embed_dim=model_config["act_embed_dim"],
        num_classes=model_config["num_classes"],
        context_size=model_config["context_size"],
        num_aug_bins=model_config["num_aug_bins"],
    ).to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Flow model has {num_params:,} parameters")

    # Compile model for faster training (before DDP wrapping)
    model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    # Wrap in DDP if multi-GPU
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True, # reduces memory and improves comm efficiency
        )

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

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            saved_config = json.load(f)
            if reflow_checkpoint is None:
                reflow_checkpoint = saved_config.get("reflow_checkpoint")
                if reflow_checkpoint and is_main:
                    print(f"Loaded reflow_checkpoint from config: {reflow_checkpoint}")
            if latent_vod is None:
                latent_vod = saved_config.get("latent_vod")
                if latent_vod and is_main:
                    print(f"Loaded latent_vod from config: {latent_vod}")

    # Load weights from encode_config.json if available
    if latent_vod:
        encode_weights = load_weights_from_encode_config(latent_vod)
        if encode_weights.get("action_weight") is not None:
            train_config["action_weight"] = encode_weights["action_weight"]
            if is_main:
                print(f"Using action_weight={encode_weights['action_weight']} from encode_config.json")
        if encode_weights.get("done_pos_weight") is not None:
            train_config["done_pos_weight"] = encode_weights["done_pos_weight"]
            if is_main:
                print(f"Using done_pos_weight={encode_weights['done_pos_weight']} from encode_config.json")

    if is_main:
        print(f"Run directory: {run_dir}")
        # Save config
        if not os.path.exists(config_path):
            config = {
                "model_config": model_config,
                "train_config": train_config,
                "reflow_checkpoint": reflow_checkpoint,
                "latent_vod": latent_vod,
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
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        total_wall_time = ckpt.get("wall_time_s", 0.0)
        
        if is_main:
            print(f"Resumed at epoch {start_epoch}")
        if world_size > 1:
            dist.barrier() # sync after checkpoint loading

    # Setup data with DistributedSampler for multi-GPU
    dataset = LatentTraceDataset(latent_vod, k=model_config["context_size"])

    sampler = DistributedSampler[Any](dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],  # per gpu
        sampler=sampler,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,  # Required for torch.compile with static shapes
    )

    if is_main:
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Batches per epoch: {len(dataloader)}")

    log_file = open(log_path, "a", buffering=1) if is_main else None
    train_start_t = time.perf_counter()

    max_aug_std = train_config["max_aug_std"]
    num_aug_bins = model_config["num_aug_bins"]
    cfg_dropout_prob = train_config["cfg_dropout_prob"]
    done_loss_weight = train_config["done_loss_weight"]
    done_pos_weight = train_config["done_pos_weight"]
    action_weight = train_config["action_weight"]

    model.train()
    epoch_iterator = tqdm(range(start_epoch, train_config["num_epochs"])) if is_main else range(start_epoch, train_config["num_epochs"])
    for epoch in epoch_iterator:
        sampler.set_epoch(epoch) 

        epoch_loss = 0.0
        epoch_done_loss = 0.0

        for batch in dataloader:
            past_frames, current_frame, actions, done_labels = batch

            past_frames = past_frames.to(device, non_blocking=True)
            current_frame = current_frame.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            done_labels = done_labels.to(device, non_blocking=True)

            B = past_frames.shape[0]
            z_target = current_frame
            z_cond = past_frames.flatten(1, 2)

            aug_level = torch.randint(0, num_aug_bins, (B,), device=device)
            aug_std = aug_level.float() / num_aug_bins * max_aug_std
            z_cond = z_cond + torch.randn_like(z_cond) * aug_std.view(B, 1, 1, 1)

            if cfg_dropout_prob > 0:
                cfg_mask = torch.rand(B, device=device) < cfg_dropout_prob
                z_cond = torch.where(cfg_mask.view(B, 1, 1, 1), torch.zeros_like(z_cond), z_cond)

            if reflow_generator is not None:
                z_0 = reflow_generator.generate(z_target, z_cond, actions, aug_level)
            else:
                z_0 = None  # sample from N(0,1) in loss function

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, done_loss = flow_matching_loss(
                    model, z_target, z_cond, actions, aug_level, z_0=z_0,
                    done_labels=done_labels, done_loss_weight=done_loss_weight,
                    action_weight=action_weight, done_pos_weight=done_pos_weight
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0) # clip on raw model
            optimizer.step()

            epoch_loss += loss.item()
            epoch_done_loss += done_loss.item()

        scheduler.step()

        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        avg_done_loss = epoch_done_loss / n_batches
        wall_time_s = total_wall_time + (time.perf_counter() - train_start_t)
        current_lr = scheduler.get_last_lr()[0]

        if is_main:
            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "done_loss": avg_done_loss,
                "lr": current_lr,
                "wall_time_s": wall_time_s,
            }
            log_file.write(json.dumps(log_entry) + "\n")

            print(f"epoch {epoch+1:4d}/{train_config['num_epochs']} loss: {avg_loss:.6f}")

        # Checkpoint saving (main process only, with barrier for sync)
        if (epoch + 1) % train_config["checkpoint_interval"] == 0:
            if world_size > 1:
                dist.barrier()

            if is_main:
                ckpt = {
                    "model": raw_model.state_dict(),  # save raw model, not DDP wrapper
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),  # save scheduler state
                    "epoch": epoch + 1,
                    "wall_time_s": wall_time_s,
                    "model_config": model_config,
                    "train_config": train_config,
                }
                torch.save(ckpt, latest_ckpt_path)
                torch.save(ckpt, os.path.join(checkpoint_dir, f"ep_{epoch+1:05d}.pt"))

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
    parser.add_argument("--latent-vod", type=str, default=None,
                        help="Path to pre-computed latent VOD directory (skips VAE encoding)")
    args = parser.parse_args()
    train(run_dir=args.run_dir, reflow_checkpoint=args.reflow, latent_vod=args.latent_vod)
