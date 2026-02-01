import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.vae import VAE
from vae.vae_data import FramesDataset

model_config = {
    "image_channels": 3,
    "hidden_channels": 32,
    "latent_channels": 4,
    "num_layers": 3,
}

train_config = {
    "lr": 1e-4,
    "num_epochs": 20,
    "batch_size": 128,
    "kl_weight": 1e-3,
    "grad_weight": 1.0,
    "bird_weight": 10.0,  # extra weight for bird pixels
    "gameover_weight": 10.0,  # extra weight for game-over sign pixels
    "log_interval": 1,
    "checkpoint_interval": 5,
    "num_workers": 4,
    "data_dir": "/home/willzhao/flappy/vod",
    "runs_dir": "/home/willzhao/flappy/diffuse/vae/runs",
}


def setup_ddp():
    if "RANK" in os.environ: # launched via torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, local_rank, world_size, device
    else: # single gpu fallback (direct python invocation)
        return 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bird_weight_mask(target, bird_weight=10.0):
    # hardcoded bird detection
    # NOTE: will mess up for end of game frames but i dont care :)
    img = (target + 1) / 2
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    bird_mask = (r > 0.8) & (g > 0.3) & (g < 0.6) & (b < 0.3) # detect orange bird body

    B, _, H, W = target.shape
    weights = torch.ones(B, 1, H, W, device=target.device)

    for i in range(B):
        mask_i = bird_mask[i]
        if mask_i.any():
            rows = mask_i.any(dim=1).nonzero(as_tuple=True)[0]
            y_min = max(0, rows.min().item() - 23)
            y_max = min(H - 1, y_min + 35)
            weights[i, 0, y_min:y_max+1, 50:101] = bird_weight
    return weights


def get_gameover_weight_mask(is_gameover, target, gameover_weight=10.0):
    # hardcoded game-over sign bounding box
    # sprite 192x42, placed at x=(288-192)//2=48, y=int(512*0.4)=204
    B, _, H, W = target.shape
    weights = torch.ones(B, 1, H, W, device=target.device)
    for i in range(B):
        if is_gameover[i]:
            weights[i, 0, 204:247, 48:241] = gameover_weight
    return weights


def weighted_l1_loss(recon, target, weights):
    # Use a true weighted mean so loss scale doesn't depend on how many
    # weighted pixels happened to appear in the batch.
    abs_err = (recon - target).abs()
    w = weights.expand_as(abs_err)  # broadcast weights across channels
    return (w * abs_err).sum() / w.sum()


def weighted_gradient_loss(recon, target, weights):
    recon_dx = recon[:, :, :, 1:] - recon[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    recon_dy = recon[:, :, 1:, :] - recon[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    # slice weights to match gradient tensor sizes
    weights_dx = weights[:, :, :, 1:]  # W-1
    weights_dy = weights[:, :, 1:, :]  # H-1

    abs_dx = (recon_dx - target_dx).abs()
    abs_dy = (recon_dy - target_dy).abs()
    w_dx = weights_dx.expand_as(abs_dx)
    w_dy = weights_dy.expand_as(abs_dy)
    grad_x = (w_dx * abs_dx).sum() / w_dx.sum()
    grad_y = (w_dy * abs_dy).sum() / w_dy.sum()
    return grad_x + grad_y

def compute_latent_statistics(model, data_dir, device, num_samples, batch_size):
    model.eval()
    frame_paths = list(Path(data_dir).glob("*/*/frames/*.png"))
    frame_paths = random.sample(frame_paths, num_samples)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    latents = []
    with torch.inference_mode():
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i+batch_size]
            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                x = transform(img).to(device)
                batch_images.append(x)
            batch_tensor = torch.stack(batch_images).to(device)
            encoded = model.encoder(batch_tensor)
            z, _, _ = model.reparameterize(encoded, sample=False)
            latents.append(z)

    latents = torch.cat(latents, dim=0)
    latent_mean = latents.mean().item()
    latent_std = latents.std().item()

    return {"latent_mean": latent_mean, "latent_std": latent_std}


def train(run_dir=None):
    rank, local_rank, world_size, device = setup_ddp()
    is_main = rank == 0

    model = VAE(
        image_channels=model_config["image_channels"],
        hidden_channels=model_config["hidden_channels"],
        latent_channels=model_config["latent_channels"],
        num_layers=model_config["num_layers"],
    ).to(device)

    if is_main:
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        print(f"Decoder has {decoder_params:,} parameters")
        print(f"Encoder has {encoder_params:,} parameters")

    # compile model (requires static shapes, drop_last=True in training)
    model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True, # reduces memory and improves comm efficiency
        )

    raw_model = model.module if world_size > 1 else model

    optimizer = AdamW(raw_model.parameters(), lr=train_config["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"])

    # setup run directory
    if is_main:
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(train_config["runs_dir"], f"vae_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

    if world_size > 1: # broadcast to all ranks for checkpoint paths
        run_dir_list = [run_dir] if is_main else [None]
        dist.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_path = os.path.join(run_dir, "training_log.jsonl")
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    config_path = os.path.join(run_dir, "config.json")

    if is_main:
        print(f"Run directory: {run_dir}")

        # save config
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(train_config, f, indent=2)

    # try to resume from checkpoint
    start_epoch = 0
    total_wall_time = 0.0

    if os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"]

        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
        raw_model.load_state_dict(state_dict) # load to raw model, not ddp wrapper
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        total_wall_time = ckpt["wall_time_s"]

        if is_main:
            print(f"resumed at epoch {start_epoch}")
        if world_size > 1:
            dist.barrier() # sync after checkpoint loading

    # setup data
    dataset = FramesDataset(train_config["data_dir"])

    sampler = DistributedSampler[Any](dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"], # per gpu
        sampler=sampler,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,  # static shapes
    )

    if is_main:
        print(f"dataset size: {len(dataset)} frames")
        print(f"batches per epoch: {len(dataloader)}")

    log_file = open(log_path, "a", buffering=1) if is_main else None
    train_start_t = time.perf_counter()

    model.train()
    epoch_iterator = tqdm(range(start_epoch, train_config["num_epochs"])) if is_main else range(start_epoch, train_config["num_epochs"])
    for epoch in epoch_iterator:
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_grad = 0.0
        epoch_kl = 0.0

        for batch, is_gameover in dataloader:
            batch = batch.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded = raw_model.encoder(batch)
                z, mean, logvar = raw_model.reparameterize(encoded, sample=True)
                recon = raw_model.decoder(z)

                # downsample if decoder outputs lower resolution
                target = batch
                if recon.shape[-2:] != batch.shape[-2:]:
                    target = F.interpolate(batch, size=recon.shape[-2:], mode='area')

                bird_weights = get_bird_weight_mask(target, bird_weight=train_config["bird_weight"])
                gameover_weights = get_gameover_weight_mask(is_gameover, target, gameover_weight=train_config["gameover_weight"])
                weights = torch.maximum(bird_weights, gameover_weights) # not doubling up 
                l1_loss = weighted_l1_loss(recon, target, weights)
                grad_loss = weighted_gradient_loss(recon, target, weights)
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

        # step scheduler and get current learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main:
            log_file.write(json.dumps({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "l1_loss": avg_l1,
                "grad_loss": avg_grad,
                "kl_loss": avg_kl,
                "lr": current_lr,
                "wall_time_s": wall_time_s,
            }) + "\n")

            print(
                f"epoch {epoch+1:4d}/{train_config['num_epochs']} | "
                f"total loss: {avg_loss:.4f} | "
                f"l1: {avg_l1:.4f} | "
                f"grad: {avg_grad:.4f} | "
                f"kl: {avg_kl:.4f}"
            )

        if (epoch + 1) % train_config["checkpoint_interval"] == 0:
            if world_size > 1:
                dist.barrier()

            if is_main:
                # compute latent statistics for checkpoint
                model_for_stats = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
                stats = compute_latent_statistics(
                    model_for_stats,
                    train_config["data_dir"],
                    device=device,
                    num_samples=1024,
                    batch_size=train_config["batch_size"],
                )
                print(f"saving checkpoint {epoch+1:05d}: latent mean: {stats['latent_mean']:.4f}, latent std: {stats['latent_std']:.4f}")

                # unwrap compiled model before saving to avoid _orig_mod prefix
                model_for_save = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model

                ckpt = {
                    "model": model_for_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "wall_time_s": wall_time_s,
                    "model_config": model_config,
                    "train_config": train_config,
                    "latent_mean": stats["latent_mean"],
                    "latent_std": stats["latent_std"],
                }
                torch.save(ckpt, latest_ckpt_path)
                torch.save(ckpt, os.path.join(checkpoint_dir, f"ep_{epoch+1:05d}.pt"))

            model.train() # back to training after compute_latent_statistics sets eval

    if is_main:
        log_file.close()
        print(f"\ntraining complete! run directory: {run_dir}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None) # resume from run directory
    args = parser.parse_args()
    train(run_dir=args.run_dir)
