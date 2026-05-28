import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any
from pathlib import Path
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


model_config = {
    "in_channels": 4, 
    "hidden_channels": 64, 
    "num_layers": 2,
    "embed_dim": 128,
    "act_embed_dim": 16,
    "num_classes": 3,
    "context_size": 8,
    "num_aug_bins": 10,
    "dynamics_dim": 3, # auxiliary dynamics head: predict [dy, vel_y, dx] from bottleneck
}

train_config = {
    "lr": 1e-4,
    "num_epochs": 1000,
    "batch_size": 4096,
    "max_aug_std": 0.5,
    "checkpoint_interval": 50,
    "num_workers": 4,

    "reflow_steps": 50,
    "cfg_dropout_prob": 0.1,
    "cfg_dropout_mode": "action",
    "cfg_null_action": None,
    "done_loss_weight": 0.1,    # weight for done head BCE loss
    "done_t_power": 4.0,        # emphasize done loss via normalized w(t)=t^4
    "filter_target_action": None, # set to 0 or 1 to only train on that target action (diagnostic)
    "dynamics_loss_weight": 0.3,  # weight for auxiliary dynamics MSE loss
    "runs_dir": "/home/willzhao/flappy/diffuse/ngen/runs",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def _load_saved_run_config(run_dir):
    if run_dir is None:
        return None
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def _apply_cli_overrides(args):
    saved_config = _load_saved_run_config(args.run_dir)
    if saved_config is not None:
        model_config.update(saved_config.get("model_config", {}))
        train_config.update(saved_config.get("train_config", {}))
        # Older runs used context-frame dropout and binary actions. Preserve that
        # behavior when resuming configs that predate action-only CFG.
        if "cfg_dropout_mode" not in saved_config.get("train_config", {}):
            train_config["cfg_dropout_mode"] = "frame"
            train_config["cfg_null_action"] = None
        if args.reflow is None:
            args.reflow = saved_config.get("reflow_checkpoint")
        if args.latent_vod is None:
            args.latent_vod = saved_config.get("latent_vod")

    model_overrides = {
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "embed_dim": args.embed_dim,
        "act_embed_dim": args.act_embed_dim,
        "num_classes": args.num_classes,
        "context_size": args.context_size,
        "num_aug_bins": args.num_aug_bins,
        "dynamics_dim": args.dynamics_dim,
    }
    for key, value in model_overrides.items():
        if value is not None:
            model_config[key] = value

    train_overrides = {
        "lr": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_aug_std": args.max_aug_std,
        "checkpoint_interval": args.checkpoint_interval,
        "num_workers": args.num_workers,
        "reflow_steps": args.reflow_steps,
        "cfg_dropout_prob": args.cfg_dropout_prob,
        "cfg_dropout_mode": args.cfg_dropout_mode,
        "cfg_null_action": args.cfg_null_action,
        "done_loss_weight": args.done_loss_weight,
        "done_t_power": args.done_t_power,
        "dynamics_loss_weight": args.dynamics_loss_weight,
    }
    for key, value in train_overrides.items():
        if value is not None:
            train_config[key] = value

    if args.filter_target_action is not None:
        train_config["filter_target_action"] = None if args.filter_target_action == "none" else int(args.filter_target_action)


def _resolve_cfg_config():
    mode = train_config.get("cfg_dropout_mode", "frame")
    if mode not in {"none", "frame", "action"}:
        raise ValueError(f"unknown cfg_dropout_mode: {mode}")
    if train_config.get("cfg_dropout_prob", 0.0) <= 0:
        mode = "none"
        train_config["cfg_dropout_mode"] = mode
    if mode == "action":
        if model_config["num_classes"] < 3:
            raise ValueError("action CFG dropout requires --num-classes >= 3 for a null action token")
        if train_config.get("cfg_null_action") is None:
            train_config["cfg_null_action"] = model_config["num_classes"] - 1
        null_action = int(train_config["cfg_null_action"])
        if null_action < 2 or null_action >= model_config["num_classes"]:
            raise ValueError(
                f"cfg_null_action={null_action} must be a non-game action id in [2, {model_config['num_classes'] - 1}]"
            )
    else:
        train_config["cfg_null_action"] = None


def load_flow_model(checkpoint_path, device): # load reflow model (no ddp, inference only)
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
        dynamics_dim=cfg.get("dynamics_dim", 0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


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


def train(latent_vod=None, run_dir=None, reflow_checkpoint=None, run_tag=None):
    rank, local_rank, world_size, device = setup_ddp()
    is_main = rank == 0
    _resolve_cfg_config()

    reflow_generator = None
    if reflow_checkpoint is not None:
        if is_main:
            print(f"loading reflow model: {reflow_checkpoint}")
        reflow_model = load_flow_model(reflow_checkpoint, device)
        reflow_generator = ReflowPairGenerator(
            reflow_model, 
            num_steps=train_config["reflow_steps"],
            cfg_scale=1.5,  # Use CFG during reflow backward integration
            num_classes=model_config["num_classes"],
            cfg_mode="auto",
            null_action_id=train_config["cfg_null_action"],
        )
        if is_main:
            print("reflow mode enabled: using pre-trained model to generate (z_0, z_1) pairs")

    model = ResUNet(
        in_channels=model_config["in_channels"],
        hidden_channels=model_config["hidden_channels"],
        num_layers=model_config["num_layers"],
        embed_dim=model_config["embed_dim"],
        act_embed_dim=model_config["act_embed_dim"],
        num_classes=model_config["num_classes"],
        context_size=model_config["context_size"],
        num_aug_bins=model_config["num_aug_bins"],
        dynamics_dim=model_config.get("dynamics_dim", 0),
    ).to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"flow model: {num_params:,} parameters")

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
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"], eta_min=1e-6)

    if is_main:
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "reflow" if reflow_checkpoint else "ngen"
            tag_suffix = f"_{run_tag}" if run_tag else ""
            run_dir = os.path.join(train_config["runs_dir"], f"{prefix}_{timestamp}{tag_suffix}")
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

    if latent_vod is None:
        raise ValueError("--latent-vod is required for new runs, or must be present in run config when resuming")

    encode_config_path = Path(latent_vod) / "encode_config.json"
    with open(encode_config_path, "r") as f:
        config = json.load(f)
    train_config["action_weight"] = config.get("action_weight")
    train_config["done_pos_weight"] = config.get("done_pos_weight")
    if is_main:
        print(f"loaded action_weight={config.get('action_weight')} from encode_config.json")
        print(f"loaded done_pos_weight={config.get('done_pos_weight')} from encode_config.json")

    if is_main:
        print(f"run directory: {run_dir}")
        print(f"model config: {json.dumps(model_config, indent=2)}")
        print(f"train config: {json.dumps(train_config, indent=2)}")
        if not os.path.exists(config_path):
            config = {
                "model_config": model_config,
                "train_config": train_config,
                "reflow_checkpoint": reflow_checkpoint,
                "latent_vod": latent_vod,
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    start_epoch = 0
    total_wall_time = 0.0

    if os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])  # load to raw model, not ddp wrapper
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        total_wall_time = ckpt.get("wall_time_s", 0.0)
        
        if is_main:
            print(f"resuming from {latest_ckpt_path} at epoch {start_epoch}")
        if world_size > 1:
            dist.barrier() # sync after checkpoint loading

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
        drop_last=True,  # required for torch.compile with static shapes
    )

    if is_main:
        print(f"dataset size: {len(dataset)} samples")
        print(f"batches per epoch: {len(dataloader)}")

    log_file = open(log_path, "a", buffering=1) if is_main else None
    train_start_t = time.perf_counter()

    model.train()
    epoch_iterator = tqdm(range(start_epoch, train_config["num_epochs"])) if is_main else range(start_epoch, train_config["num_epochs"])
    for epoch in epoch_iterator:
        sampler.set_epoch(epoch) 

        epoch_loss = 0.0
        epoch_flow_loss = 0.0
        epoch_done_loss = 0.0
        epoch_dynamics_loss = 0.0

        for batch in dataloader:
            past_frames, current_frame, actions, done_labels, game_states = batch

            past_frames = past_frames.to(device, non_blocking=True)
            current_frame = current_frame.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            done_labels = done_labels.to(device, non_blocking=True)
            game_states = game_states.to(device, non_blocking=True)

            if train_config["filter_target_action"] is not None:
                mask = actions[:, -1] == train_config["filter_target_action"]
                if mask.sum() == 0:
                    continue
                past_frames, current_frame = past_frames[mask], current_frame[mask]
                actions, done_labels = actions[mask], done_labels[mask]
                game_states = game_states[mask]

            B = past_frames.shape[0]
            z_target = current_frame
            z_cond = past_frames.flatten(1, 2)
            model_actions = actions

            aug_level = torch.randint(0, model_config["num_aug_bins"], (B,), device=device)
            aug_std = aug_level.float() / model_config["num_aug_bins"] * train_config["max_aug_std"]
            z_cond = z_cond + torch.randn_like(z_cond) * aug_std.view(B, 1, 1, 1)

            if train_config["cfg_dropout_prob"] > 0:
                cfg_mask = torch.rand(B, device=device) < train_config["cfg_dropout_prob"]
                if train_config["cfg_dropout_mode"] == "frame":
                    z_cond = torch.where(cfg_mask.view(B, 1, 1, 1), torch.zeros_like(z_cond), z_cond)
                elif train_config["cfg_dropout_mode"] == "action":
                    null_actions = torch.full_like(actions, int(train_config["cfg_null_action"]))
                    model_actions = torch.where(cfg_mask.view(B, 1), null_actions, actions)

            if reflow_generator is not None:
                z_0 = reflow_generator.generate(z_target, z_cond, model_actions, aug_level)
            else:
                z_0 = None  # sample from N(0,1) in loss function

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, flow_loss, done_loss, dynamics_loss = flow_matching_loss(
                    model, z_target, z_cond, model_actions, aug_level, z_0=z_0,
                    done_labels=done_labels, done_loss_weight=train_config["done_loss_weight"],
                    action_weight=train_config["action_weight"], done_pos_weight=train_config["done_pos_weight"],
                    done_t_power=train_config["done_t_power"],
                    game_states=game_states, dynamics_loss_weight=train_config["dynamics_loss_weight"],
                    loss_actions=actions,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0) # clip on raw model
            optimizer.step()

            epoch_loss += loss.item()
            epoch_flow_loss += flow_loss.item()
            epoch_done_loss += done_loss.item()
            epoch_dynamics_loss += dynamics_loss.item()
        scheduler.step()

        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        avg_flow_loss = epoch_flow_loss / n_batches
        avg_done_loss = epoch_done_loss / n_batches
        avg_dynamics_loss = epoch_dynamics_loss / n_batches
        wall_time_s = total_wall_time + (time.perf_counter() - train_start_t)
        current_lr = scheduler.get_last_lr()[0]

        if is_main:
            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "flow_loss": avg_flow_loss,
                "done_loss": avg_done_loss,
                "dynamics_loss": avg_dynamics_loss,
                "lr": current_lr,
                "wall_time_s": wall_time_s,
            }
            log_file.write(json.dumps(log_entry) + "\n")

            print(f"epoch {epoch+1:4d}/{train_config['num_epochs']} | "
                  f"loss: {avg_loss:.4f} | "
                  f"flow: {avg_flow_loss:.4f} | "
                  f"done: {avg_done_loss:.4f} | "
                  f"dyn: {avg_dynamics_loss:.4f}")

        # checkpoint saving (main process only, with barrier for sync)
        if (epoch + 1) % train_config["checkpoint_interval"] == 0:
            if world_size > 1:
                dist.barrier()

            if is_main:
                ckpt = {
                    "model": raw_model.state_dict(),  # save raw model, not ddp wrapper
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),  # save scheduler
                    "epoch": epoch + 1,
                    "wall_time_s": wall_time_s,
                    "model_config": model_config,
                    "train_config": train_config,
                }
                torch.save(ckpt, latest_ckpt_path)
                torch.save(ckpt, os.path.join(checkpoint_dir, f"ep_{epoch+1:05d}.pt"))

    if is_main:
        log_file.close()
        print(f"\ntraining complete! run directory: {run_dir}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-vod", type=str, default=None) 
    parser.add_argument("--run-dir", type=str, default=None) # resume from run directory
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--reflow", type=str, default=None) # pre-trained checkpoint for reflow 
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--act-embed-dim", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--num-aug-bins", type=int, default=None)
    parser.add_argument("--dynamics-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-aug-std", type=float, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--reflow-steps", type=int, default=None)
    parser.add_argument("--cfg-dropout-prob", type=float, default=None)
    parser.add_argument("--cfg-dropout-mode", choices=["none", "frame", "action"], default=None)
    parser.add_argument("--cfg-null-action", type=int, default=None)
    parser.add_argument("--done-loss-weight", type=float, default=None)
    parser.add_argument("--done-t-power", type=float, default=None)
    parser.add_argument("--dynamics-loss-weight", type=float, default=None)
    parser.add_argument("--filter-target-action", choices=["none", "0", "1"], default=None)
    args = parser.parse_args()
    _apply_cli_overrides(args)
    train(latent_vod=args.latent_vod, run_dir=args.run_dir, reflow_checkpoint=args.reflow, run_tag=args.run_tag)
