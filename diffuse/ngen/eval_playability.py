"""Inference-only playability probes for Flappy world-model checkpoints.

This script does not train. It samples recorded VOD windows, encodes frames
through a frozen VAE, and measures where closed-loop generation loses
consistency: one-step prediction, action sensitivity, noise sensitivity, and
autoregressive rollout drift.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffuse.ngen.sampler import euler_sample  # noqa: E402
from diffuse.nn.embedding import TimeEmbedding  # noqa: E402
from diffuse.nn.resblock import DownResBlock, ResBlock, UpResBlock, get_groups  # noqa: E402
from diffuse.nn.resunet import ResUNet  # noqa: E402
from diffuse.nn.vae import VAE  # noqa: E402


class LegacyTimeAugResBlock(nn.Module):
    """Older spatial-action checkpoints condition ResBlocks on time + aug only."""

    def __init__(self, fan_in: int, fan_out: int, embed_dim: int, groups: int = 16):
        super().__init__()
        self.gn1 = nn.GroupNorm(get_groups(fan_in, groups), fan_in, affine=True)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(fan_in, fan_out, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(get_groups(fan_out, groups), fan_out, affine=False)
        self.proj = nn.Linear(embed_dim * 2, fan_out * 2)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(fan_out, fan_out, kernel_size=3, padding=1)
        self.act3 = nn.SiLU()
        self.skip_conv = nn.Conv2d(fan_in, fan_out, kernel_size=1) if fan_in != fan_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, aug_emb: torch.Tensor) -> torch.Tensor:
        res = self.skip_conv(x)
        x = self.gn2(self.conv1(self.act1(self.gn1(x))))
        emb = torch.cat([t_emb, aug_emb], dim=1)
        scale, shift = self.proj(emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.conv2(self.act2(x))
        return self.act3(x + res)


class LegacyTimeAugDownResBlock(nn.Module):
    def __init__(self, fan_in: int, fan_out: int, embed_dim: int):
        super().__init__()
        self.res = LegacyTimeAugResBlock(fan_in, fan_out, embed_dim)
        self.down = nn.Conv2d(fan_out, fan_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, aug_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res(x, t_emb, aug_emb)
        return self.down(x), x


class LegacyTimeAugUpResBlock(nn.Module):
    def __init__(self, fan_in: int, fan_out: int, embed_dim: int):
        super().__init__()
        self.up = nn.Conv2d(fan_in, fan_out, kernel_size=3, padding=1)
        self.res = LegacyTimeAugResBlock(fan_out * 2, fan_out, embed_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor, aug_emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb, aug_emb)


class LegacySpatialActionResUNet(nn.Module):
    """Checkpoint layout used by spatial-action-only runs such as 20260207."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        embed_dim: int,
        act_embed_dim: int,
        num_classes: int,
        context_size: int,
        num_aug_bins: int,
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(embed_dim=embed_dim)
        self.aug_embedding = nn.Embedding(num_aug_bins, embed_dim)
        self.aug_act = nn.SiLU()
        self.class_embedding = nn.Embedding(num_classes, act_embed_dim)
        self.class_spatial_proj = nn.Linear(act_embed_dim, act_embed_dim)

        first_in_channels = in_channels * (context_size + 1) + act_embed_dim
        self.down_blocks = nn.ModuleList([LegacyTimeAugDownResBlock(first_in_channels, hidden_channels, embed_dim)])
        for i in range(num_layers - 1):
            self.down_blocks.append(LegacyTimeAugDownResBlock(hidden_channels * 2**i, hidden_channels * 2 ** (i + 1), embed_dim))

        self.bot = LegacyTimeAugResBlock(hidden_channels * 2 ** (num_layers - 1), hidden_channels * 2**num_layers, embed_dim)
        bot_channels = hidden_channels * 2**num_layers
        self.bot_attn_norm = nn.GroupNorm(1, bot_channels)
        self.bot_attn = nn.MultiheadAttention(bot_channels, num_heads=8, batch_first=True)
        self.done_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bot_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.up_blocks.append(LegacyTimeAugUpResBlock(hidden_channels * 2 ** (num_layers - i), hidden_channels * 2 ** (num_layers - i - 1), embed_dim))
        self.out_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z_cond: torch.Tensor, c: torch.Tensor, aug_level: torch.Tensor):
        b, _, h, w = x.shape
        target_action = c[:, -1]
        action_spatial = self.class_spatial_proj(self.class_embedding(target_action))
        action_spatial = action_spatial.unsqueeze(-1).unsqueeze(-1).expand(b, -1, h, w)
        x = torch.cat([x, z_cond, action_spatial], dim=1)

        t_emb = self.time_embedding(t)
        aug_emb = self.aug_act(self.aug_embedding(aug_level))

        skips = []
        for down in self.down_blocks:
            x, skip = down(x, t_emb, aug_emb)
            skips.append(skip)
        x = self.bot(x, t_emb, aug_emb)

        b_attn, c_attn, h_attn, w_attn = x.shape
        x_flat = self.bot_attn_norm(x).reshape(b_attn, c_attn, -1).permute(0, 2, 1)
        attn_out, _ = self.bot_attn(x_flat, x_flat, x_flat)
        x = x + attn_out.permute(0, 2, 1).reshape(b_attn, c_attn, h_attn, w_attn)
        done_logit = self.done_head(x.detach())

        for up in self.up_blocks:
            x = up(x, skips.pop(), t_emb, aug_emb)
        return self.out_conv(x), done_logit


class LegacyAdaGNActionResUNet(nn.Module):
    """Checkpoint layout used before spatial action channels were added."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        embed_dim: int,
        act_embed_dim: int,
        num_classes: int,
        context_size: int,
        num_aug_bins: int,
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(embed_dim=embed_dim)
        self.aug_embedding = nn.Embedding(num_aug_bins, embed_dim)
        self.aug_act = nn.SiLU()
        self.class_embedding = nn.Embedding(num_classes, act_embed_dim)
        self.class_proj = nn.Sequential(nn.SiLU(), nn.Linear((context_size + 1) * act_embed_dim, embed_dim))

        first_in_channels = in_channels * (context_size + 1)
        self.down_blocks = nn.ModuleList([DownResBlock(first_in_channels, hidden_channels, embed_dim)])
        for i in range(num_layers - 1):
            self.down_blocks.append(DownResBlock(hidden_channels * 2**i, hidden_channels * 2 ** (i + 1), embed_dim))

        self.bot = ResBlock(hidden_channels * 2 ** (num_layers - 1), hidden_channels * 2**num_layers, embed_dim)
        bot_channels = hidden_channels * 2**num_layers
        self.bot_attn_norm = nn.GroupNorm(1, bot_channels)
        self.bot_attn = nn.MultiheadAttention(bot_channels, num_heads=8, batch_first=True)
        self.done_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bot_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.up_blocks.append(UpResBlock(hidden_channels * 2 ** (num_layers - i), hidden_channels * 2 ** (num_layers - i - 1), embed_dim))
        self.out_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z_cond: torch.Tensor, c: torch.Tensor, aug_level: torch.Tensor):
        x = torch.cat([x, z_cond], dim=1)
        t_emb = self.time_embedding(t)
        aug_emb = self.aug_act(self.aug_embedding(aug_level))
        c_emb = self.class_proj(self.class_embedding(c).flatten(1))

        skips = []
        for down in self.down_blocks:
            x, skip = down(x, t_emb, aug_emb, c_emb)
            skips.append(skip)
        x = self.bot(x, t_emb, aug_emb, c_emb)

        b_attn, c_attn, h_attn, w_attn = x.shape
        x_flat = self.bot_attn_norm(x).reshape(b_attn, c_attn, -1).permute(0, 2, 1)
        attn_out, _ = self.bot_attn(x_flat, x_flat, x_flat)
        x = x + attn_out.permute(0, 2, 1).reshape(b_attn, c_attn, h_attn, w_attn)
        done_logit = self.done_head(x.detach())

        for up in self.up_blocks:
            x = up(x, skips.pop(), t_emb, aug_emb, c_emb)
        return self.out_conv(x), done_logit


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    frame_files: tuple[Path, ...]
    actions: dict[int, int]
    done: dict[int, bool]


@dataclass(frozen=True)
class Window:
    run: RunData
    start: int


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        return p
    # Checkpoints/configs copied from the cluster often contain /home/willzhao/flappy.
    marker = "/flappy/"
    s = str(path)
    if marker in s:
        candidate = ROOT / s.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return p


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_vae(checkpoint_path: Path, device: torch.device, latent_stats_path: Path | None) -> tuple[VAE, float, float]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    vae = VAE(
        image_channels=cfg["image_channels"],
        hidden_channels=cfg["hidden_channels"],
        latent_channels=cfg["latent_channels"],
        num_layers=cfg["num_layers"],
    ).to(device)
    vae.load_state_dict(strip_compile_prefix(ckpt["model"]))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    latent_mean = ckpt.get("latent_mean")
    latent_std = ckpt.get("latent_std")
    if (latent_mean is None or latent_std is None) and latent_stats_path is not None and latent_stats_path.exists():
        with latent_stats_path.open() as f:
            stats = json.load(f)
        latent_mean = stats.get("latent_mean")
        latent_std = stats.get("latent_std")
    if latent_mean is None or latent_std is None:
        raise ValueError("VAE latent_mean/latent_std missing; pass --latent-stats with encode_config.json")
    return vae, float(latent_mean), float(latent_std)


def load_flow(checkpoint_path: Path, device: torch.device) -> tuple[ResUNet, dict, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_config"])
    cfg["context_size"] = int(cfg.get("context_size", cfg.get("context_frames")))
    state_dict = strip_compile_prefix(ckpt["model"])

    common = {
        "in_channels": cfg["in_channels"],
        "hidden_channels": cfg["hidden_channels"],
        "num_layers": cfg["num_layers"],
        "embed_dim": cfg["embed_dim"],
        "act_embed_dim": cfg.get("act_embed_dim", cfg["embed_dim"]),
        "num_classes": cfg["num_classes"],
        "context_size": cfg["context_size"],
        "num_aug_bins": cfg["num_aug_bins"],
    }
    if any(k.startswith("action_adagn_proj.") for k in state_dict):
        model = ResUNet(**common, dynamics_dim=cfg.get("dynamics_dim", 0)).to(device)
    elif "class_spatial_proj.weight" in state_dict:
        model = LegacySpatialActionResUNet(**common).to(device)
    elif any(k.startswith("class_proj.") for k in state_dict):
        model = LegacyAdaGNActionResUNet(**common).to(device)
    else:
        raise ValueError(f"unsupported ResUNet checkpoint layout: {checkpoint_path}")

    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, dict(cfg), dict(ckpt.get("train_config", {}))


def load_runs(vod_dir: Path, min_frames: int, max_runs: int, seed: int) -> list[RunData]:
    runs: list[RunData] = []
    for run_dir in sorted(vod_dir.glob("*/*/")):
        frames_dir = run_dir / "frames"
        info_file = run_dir / "run_info.jsonl"
        if not frames_dir.exists() or not info_file.exists():
            continue
        frame_files = tuple(sorted(frames_dir.glob("*.png")))
        if len(frame_files) < min_frames:
            continue

        actions: dict[int, int] = {}
        done: dict[int, bool] = {}
        with info_file.open() as f:
            for line in f:
                rec = json.loads(line)
                if "step" not in rec:
                    continue
                step = int(rec["step"])
                actions[step] = int(rec.get("action", 0))
                done[step] = bool(rec.get("terminated", False))
        if len(actions) < min_frames:
            continue
        runs.append(RunData(run_dir=run_dir, frame_files=frame_files, actions=actions, done=done))

    rng = random.Random(seed)
    rng.shuffle(runs)
    if max_runs > 0:
        runs = runs[:max_runs]
    return runs


def choose_windows(
    runs: Iterable[RunData],
    k: int,
    rollout_len: int,
    per_action: int,
    seed: int,
) -> list[Window]:
    rng = random.Random(seed)
    by_action: dict[int, list[Window]] = {0: [], 1: []}

    for run in runs:
        n = len(run.frame_files)
        upper = n - rollout_len
        if upper <= k:
            continue
        candidates = list(range(k, upper))
        rng.shuffle(candidates)
        for start in candidates[: max(64, per_action * 8)]:
            action = run.actions.get(start)
            if action in by_action:
                by_action[action].append(Window(run=run, start=start))

    windows: list[Window] = []
    for action in (0, 1):
        rng.shuffle(by_action[action])
        windows.extend(by_action[action][:per_action])
    rng.shuffle(windows)
    return windows


def image_batch(paths: list[Path]) -> torch.Tensor:
    tensors = []
    for path in paths:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(x * 2.0 - 1.0)
    return torch.stack(tensors, dim=0)


@torch.inference_mode()
def encode_paths(
    vae: VAE,
    paths: list[Path],
    latent_mean: float,
    latent_std: float,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    latents = []
    for i in range(0, len(paths), batch_size):
        batch = image_batch(paths[i : i + batch_size]).to(device)
        encoded = vae.encoder(batch)
        z, _, _ = vae.reparameterize(encoded, sample=False)
        z = (z - latent_mean) / latent_std
        latents.append(z.detach().cpu())
    return torch.cat(latents, dim=0)


def z0_like(mode: str, shape: torch.Size, device: torch.device, prev: torch.Tensor | None = None) -> torch.Tensor:
    if mode in {"random", "corr"}:
        return torch.randn(shape, device=device)
    if mode == "zero":
        return torch.zeros(shape, device=device)
    if mode == "prev":
        if prev is None:
            raise ValueError("prev z0 mode requires a previous latent")
        return prev.clone()
    if mode.startswith("prev_noise_"):
        if prev is None:
            raise ValueError("prev_noise z0 mode requires a previous latent")
        scale = float(mode.removeprefix("prev_noise_"))
        return prev + scale * torch.randn(shape, device=device)
    raise ValueError(f"unknown z0 mode: {mode}")


def corr_noise(prev_noise: torch.Tensor | None, shape: torch.Size, device: torch.device, rho: float = 0.95) -> torch.Tensor:
    eps = torch.randn(shape, device=device)
    if prev_noise is None:
        return eps
    return rho * prev_noise + math.sqrt(max(0.0, 1.0 - rho * rho)) * eps


def mean_abs(x: torch.Tensor) -> float:
    return float(x.abs().mean().item())


def mean_sq(x: torch.Tensor) -> float:
    return float((x * x).mean().item())


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "abs_mean": float(x.abs().mean().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "clamp4_frac": float((x.abs() >= 3.99).float().mean().item()),
    }


@torch.inference_mode()
def done_prob(model: ResUNet, z: torch.Tensor, z_cond: torch.Tensor, actions: torch.Tensor, aug_level: torch.Tensor) -> float:
    t = torch.ones((z.shape[0],), device=z.device)
    _, done_logit, *_ = model(z, t, z_cond=z_cond, c=actions, aug_level=aug_level)
    return float(torch.sigmoid(done_logit).mean().item())


def parse_variants(spec: str) -> list[tuple[str, float]]:
    variants = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        mode, cfg = raw.split(":", 1)
        variants.append((mode, float(cfg)))
    return variants


@torch.inference_mode()
def run_one_step_probe(
    model: ResUNet,
    windows: list[Window],
    encoded: dict[tuple[Path, int], torch.Tensor],
    k: int,
    num_steps: int,
    variants: list[tuple[str, float]],
    noise_trials: int,
    device: torch.device,
) -> dict:
    aug_level = torch.zeros((1,), dtype=torch.long, device=device)
    results: dict[str, list[dict[str, float]]] = {f"{m}:cfg{c:g}": [] for m, c in variants}

    for window in windows:
        run = window.run
        step = window.start
        context = torch.stack([encoded[(run.run_dir, i)] for i in range(step - k, step)], dim=0)
        target = encoded[(run.run_dir, step)]
        prev = context[-1]

        z_cond = context.unsqueeze(0).flatten(1, 2).to(device)
        z_target = target.unsqueeze(0).to(device)
        z_prev = prev.unsqueeze(0).to(device)
        true_delta = mean_abs(z_target - z_prev)
        actions_base = [run.actions.get(i, 0) for i in range(step - k, step + 1)]
        actions = torch.tensor([actions_base], dtype=torch.long, device=device)
        actions_alt_list = list(actions_base)
        actions_alt_list[-1] = 1 - actions_alt_list[-1]
        actions_alt = torch.tensor([actions_alt_list], dtype=torch.long, device=device)

        for mode, cfg_scale in variants:
            key = f"{mode}:cfg{cfg_scale:g}"
            z0 = z0_like(mode, z_target.shape, device, prev=z_prev)
            pred = euler_sample(
                model,
                z0,
                z_cond,
                actions,
                aug_level,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
            )
            pred_alt = euler_sample(
                model,
                z0.clone(),
                z_cond,
                actions_alt,
                aug_level,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
            )

            noise_preds = []
            if mode == "random":
                for _ in range(noise_trials):
                    noise_preds.append(
                        euler_sample(
                            model,
                            torch.randn_like(z_target),
                            z_cond,
                            actions,
                            aug_level,
                            cfg_scale=cfg_scale,
                            num_steps=num_steps,
                        )
                    )
            noise_std = 0.0
            if len(noise_preds) >= 2:
                noise_std = float(torch.stack(noise_preds, dim=0).std(dim=0).mean().item())

            rec = {
                "target_action": float(actions_base[-1]),
                "done": float(run.done.get(step, False)),
                "true_delta_mae": true_delta,
                "pred_target_mae": mean_abs(pred - z_target),
                "pred_target_mse": mean_sq(pred - z_target),
                "pred_delta_mae": mean_abs(pred - z_prev),
                "action_diff_mae": mean_abs(pred - pred_alt),
                "action_diff_over_true_delta": mean_abs(pred - pred_alt) / max(true_delta, 1e-8),
                "noise_std": noise_std,
                "noise_std_over_true_delta": noise_std / max(true_delta, 1e-8),
                "done_prob_target": done_prob(model, z_target, z_cond, actions, aug_level),
                "done_prob_pred": done_prob(model, pred, z_cond, actions, aug_level),
                **{f"pred_{name}": value for name, value in tensor_stats(pred).items()},
            }
            results[key].append(rec)

    return {key: summarize_records_with_action_groups(records) for key, records in results.items()}


@torch.inference_mode()
def run_rollout_probe(
    model: ResUNet,
    windows: list[Window],
    encoded: dict[tuple[Path, int], torch.Tensor],
    k: int,
    rollout_len: int,
    num_steps: int,
    variants: list[tuple[str, float]],
    device: torch.device,
) -> dict:
    aug_level = torch.zeros((1,), dtype=torch.long, device=device)
    results: dict[str, list[dict[str, float]]] = {f"{m}:cfg{c:g}": [] for m, c in variants}

    for window in windows:
        run = window.run
        start = window.start

        for mode, cfg_scale in variants:
            key = f"{mode}:cfg{cfg_scale:g}"
            frame_buffer = [
                encoded[(run.run_dir, i)].unsqueeze(0).to(device)
                for i in range(start - k, start)
            ]
            noise_state: torch.Tensor | None = None

            step_records = []
            for offset in range(rollout_len):
                step = start + offset
                target = encoded[(run.run_dir, step)].unsqueeze(0).to(device)
                prev = frame_buffer[-1]
                z_cond = torch.cat(frame_buffer, dim=1)
                action_list = [run.actions.get(i, 0) for i in range(step - k, step + 1)]
                actions = torch.tensor([action_list], dtype=torch.long, device=device)

                if mode == "corr":
                    noise_state = corr_noise(noise_state, target.shape, device)
                    z0 = noise_state
                else:
                    z0 = z0_like(mode, target.shape, device, prev=prev)

                pred = euler_sample(
                    model,
                    z0,
                    z_cond,
                    actions,
                    aug_level,
                    cfg_scale=cfg_scale,
                    num_steps=num_steps,
                )

                step_records.append(
                    {
                        "offset": float(offset),
                        "target_action": float(action_list[-1]),
                        "done": float(run.done.get(step, False)),
                        "mae": mean_abs(pred - target),
                        "mse": mean_sq(pred - target),
                        "gen_delta_mae": mean_abs(pred - prev),
                        "true_delta_mae": mean_abs(target - prev),
                        "clamp4_frac": float((pred.abs() >= 3.99).float().mean().item()),
                        "done_prob_pred": done_prob(model, pred, z_cond, actions, aug_level),
                        "done_prob_target": done_prob(model, target, z_cond, actions, aug_level),
                    }
                )

                frame_buffer.pop(0)
                frame_buffer.append(pred.detach())

            summary = summarize_records(step_records)
            summary["final_mae"] = step_records[-1]["mae"]
            summary["final_mse"] = step_records[-1]["mse"]
            summary["mae_slope_per_frame"] = slope([r["mae"] for r in step_records])
            summary["mse_slope_per_frame"] = slope([r["mse"] for r in step_records])
            results[key].append(summary)

    return {key: summarize_records(records) for key, records in results.items()}


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = np.arange(len(values), dtype=np.float64)
    ys = np.asarray(values, dtype=np.float64)
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((xs - x_mean) * (ys - y_mean)).sum() / denom)


def summarize_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {"count": 0.0}
    keys = sorted({k for rec in records for k in rec})
    out: dict[str, float] = {"count": float(len(records))}
    for key in keys:
        vals = [rec[key] for rec in records if key in rec and math.isfinite(float(rec[key]))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_p50"] = float(np.percentile(arr, 50))
        out[f"{key}_p90"] = float(np.percentile(arr, 90))
    return out


def summarize_records_with_action_groups(records: list[dict[str, float]]) -> dict[str, float]:
    out = summarize_records(records)
    for action in (0, 1):
        grouped = [rec for rec in records if int(rec.get("target_action", -1)) == action]
        grouped_summary = summarize_records(grouped)
        out.update({f"action{action}_{key}": value for key, value in grouped_summary.items()})
    return out


def collect_needed_indices(windows: list[Window], k: int, rollout_len: int) -> dict[Path, set[int]]:
    needed: dict[Path, set[int]] = {}
    for window in windows:
        run = window.run
        indices = needed.setdefault(run.run_dir, set())
        for i in range(window.start - k, window.start + rollout_len):
            indices.add(i)
    return needed


def encode_needed(
    vae: VAE,
    windows: list[Window],
    k: int,
    rollout_len: int,
    latent_mean: float,
    latent_std: float,
    device: torch.device,
    batch_size: int,
) -> dict[tuple[Path, int], torch.Tensor]:
    encoded: dict[tuple[Path, int], torch.Tensor] = {}
    needed = collect_needed_indices(windows, k, rollout_len)
    for run_dir, indices_set in needed.items():
        run = next(w.run for w in windows if w.run.run_dir == run_dir)
        indices = sorted(indices_set)
        paths = [run.frame_files[i] for i in indices]
        latents = encode_paths(vae, paths, latent_mean, latent_std, device=device, batch_size=batch_size)
        for index, latent in zip(indices, latents):
            encoded[(run_dir, index)] = latent
    return encoded


def markdown_summary(metrics: dict) -> str:
    lines = [
        "# Playability Eval",
        "",
        f"- NGEN: `{metrics['ngen_checkpoint']}`",
        f"- VAE: `{metrics['vae_checkpoint']}`",
        f"- Device: `{metrics['device']}`",
        f"- Windows: {metrics['num_windows']}",
        f"- Rollout length: {metrics['rollout_len']}",
        f"- Euler steps: {metrics['num_steps']}",
        "",
        "## One-Step",
        "",
    ]
    for variant, vals in metrics["one_step"].items():
        lines.append(f"### {variant}")
        lines.append(f"- target MAE: {vals.get('pred_target_mae_mean', float('nan')):.4f}")
        lines.append(f"- real frame delta MAE: {vals.get('true_delta_mae_mean', float('nan')):.4f}")
        lines.append(f"- generated frame delta MAE: {vals.get('pred_delta_mae_mean', float('nan')):.4f}")
        lines.append(f"- action diff / real delta: {vals.get('action_diff_over_true_delta_mean', float('nan')):.3f}")
        lines.append(f"- action diff / real delta, no-flap targets: {vals.get('action0_action_diff_over_true_delta_mean', float('nan')):.3f}")
        lines.append(f"- action diff / real delta, flap targets: {vals.get('action1_action_diff_over_true_delta_mean', float('nan')):.3f}")
        lines.append(f"- noise std / real delta: {vals.get('noise_std_over_true_delta_mean', float('nan')):.3f}")
        lines.append(f"- clamp frac: {vals.get('pred_clamp4_frac_mean', float('nan')):.4f}")
        lines.append("")

    lines.append("## Rollout")
    lines.append("")
    for variant, vals in metrics["rollout"].items():
        lines.append(f"### {variant}")
        lines.append(f"- mean MAE: {vals.get('mae_mean_mean', float('nan')):.4f}")
        lines.append(f"- final MAE: {vals.get('final_mae_mean', float('nan')):.4f}")
        lines.append(f"- MAE slope/frame: {vals.get('mae_slope_per_frame_mean', float('nan')):.5f}")
        lines.append(f"- generated delta / true delta: {vals.get('gen_delta_mae_mean_mean', float('nan')) / max(vals.get('true_delta_mae_mean_mean', 1e-8), 1e-8):.3f}")
        lines.append(f"- clamp frac: {vals.get('clamp4_frac_mean_mean', float('nan')):.4f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngen-checkpoint", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--vod-dir", type=str, default=str(ROOT / "vod"))
    parser.add_argument("--latent-stats", type=str, default=str(ROOT / "latent-vod" / "encode_config.json"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "eval_runs"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=240)
    parser.add_argument("--samples-per-action", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--rollout-len", type=int, default=30)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--noise-trials", type=int, default=3)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument(
        "--variants",
        type=str,
        default="random:1.0,random:1.5,zero:1.0,corr:1.0",
        help="comma-separated z0_mode:cfg_scale entries. Modes: random, zero, corr, prev, prev_noise_X",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    ngen_checkpoint = resolve_path(args.ngen_checkpoint)
    vae_checkpoint = resolve_path(args.vae_checkpoint)
    latent_stats = resolve_path(args.latent_stats)
    vod_dir = resolve_path(args.vod_dir)
    if ngen_checkpoint is None or not ngen_checkpoint.exists():
        raise FileNotFoundError(args.ngen_checkpoint)
    if vae_checkpoint is None or not vae_checkpoint.exists():
        raise FileNotFoundError(args.vae_checkpoint)
    if vod_dir is None or not vod_dir.exists():
        raise FileNotFoundError(args.vod_dir)

    t0 = time.time()
    model, model_cfg, train_cfg = load_flow(ngen_checkpoint, device)
    vae, latent_mean, latent_std = load_vae(vae_checkpoint, device, latent_stats)
    k = int(model_cfg["context_size"])
    variants = parse_variants(args.variants)

    min_frames = k + args.rollout_len + 1
    runs = load_runs(vod_dir, min_frames=min_frames, max_runs=args.max_runs, seed=args.seed)
    windows = choose_windows(
        runs,
        k=k,
        rollout_len=args.rollout_len,
        per_action=args.samples_per_action,
        seed=args.seed,
    )
    if not windows:
        raise RuntimeError("no usable eval windows found")

    encoded = encode_needed(
        vae,
        windows,
        k=k,
        rollout_len=args.rollout_len,
        latent_mean=latent_mean,
        latent_std=latent_std,
        device=device,
        batch_size=args.encode_batch_size,
    )

    one_step = run_one_step_probe(
        model,
        windows,
        encoded,
        k=k,
        num_steps=args.num_steps,
        variants=variants,
        noise_trials=args.noise_trials,
        device=device,
    )

    rollout_windows = windows[: max(1, args.rollouts)]
    rollout = run_rollout_probe(
        model,
        rollout_windows,
        encoded,
        k=k,
        rollout_len=args.rollout_len,
        num_steps=args.num_steps,
        variants=variants,
        device=device,
    )

    out_dir = Path(args.out_dir) / time.strftime("playability_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "ngen_checkpoint": str(ngen_checkpoint),
        "vae_checkpoint": str(vae_checkpoint),
        "vod_dir": str(vod_dir),
        "device": str(device),
        "model_config": model_cfg,
        "train_config": train_cfg,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "num_windows": len(windows),
        "num_rollouts": len(rollout_windows),
        "rollout_len": args.rollout_len,
        "num_steps": args.num_steps,
        "variants": [f"{m}:cfg{c:g}" for m, c in variants],
        "elapsed_s": time.time() - t0,
        "one_step": one_step,
        "rollout": rollout,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(markdown_summary(metrics), encoding="utf-8")

    print(markdown_summary(metrics))
    print(f"\nwrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
