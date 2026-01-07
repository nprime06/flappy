"""
RL-friendly runner utilities for Flappy Bird (Gymnasium).

This project uses `flappy-bird-gymnasium` (a maintained Gymnasium environment):
  - https://github.com/markub3327/flappy-bird-gymnasium

Primary API:
  - `run_episode(...)`: run one episode with a provided agent, save metrics, optionally save frames/video.

Agent interface:
  - Either pass a callable: `action = agent(obs)`
  - Or an object with `act(obs)` method: `action = agent.act(obs)`
"""

# 0–2: nearest pipe (pipe_x, gap_top_y, gap_bottom_y)
# 3–5: next pipe (pipe_x, gap_top_y, gap_bottom_y)
# 6–8: next-next pipe (pipe_x, gap_top_y, gap_bottom_y)
# 9: bird y
# 10: bird vel_y
# 11: bird rot

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

import gymnasium as gym
import imageio.v3 as iio
import imageio

# Registers "FlappyBird-v0", "FlappyBird-rgb-v0", etc. with gymnasium.
import flappy_bird_gymnasium  # noqa: F401


class Agent(Protocol):
    def act(self, processed_obs: Any) -> int: ...


AgentLike = Union[Agent, Callable[[Any], int]]


@dataclass(frozen=True)
class RunConfig:
    env_id: str = "FlappyBird-v0"
    seed: Optional[int] = None
    max_steps: Optional[int] = None

    use_lidar: bool = False
    disable_env_checker: bool = True

    out_dir: Optional[str] = "runs"  # None = no output
    run_name: Optional[str] = None  # default is timestamp
    save_run_info: bool = True
    save_frames: bool = True
    save_video: bool = True
    video_fps: int = 30


@dataclass
class RunResult:
    config: RunConfig
    run_dir: Optional[str]

    # Episode summary
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    score: Optional[int]
    wall_time_s: float

    frames_dir: Optional[str] = None
    video_path: Optional[str] = None
    metrics_path: Optional[str] = None
    run_info_path: Optional[str] = None


def _timestamp_run_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _resolve_run_dir(cfg: RunConfig) -> Optional[Path]:
    if cfg.out_dir is None:
        return None
    run_name = cfg.run_name or _timestamp_run_name()
    return Path(cfg.out_dir).expanduser().resolve() / run_name

def _process_obs(obs: Any) -> Any:
    nearest_pipe_x = obs[0]
    nearest_gap_top_y = obs[1]
    nearest_gap_bottom_y = obs[2]
    bird_y = obs[9]
    bird_vel_y = obs[10]
    gap_middle_y = (nearest_gap_top_y + nearest_gap_bottom_y) / 2

    dy = (gap_middle_y - bird_y - 0.01) / 0.11
    dx = (nearest_pipe_x - 0.1) / 0.3
    return np.array([dy, bird_vel_y, dx])


def _agent_action(agent: AgentLike, processed_obs: Any) -> int:
    if hasattr(agent, "act"):
        # mypy/pyright: runtime check
        return int(getattr(agent, "act")(processed_obs))
    return int(agent(processed_obs))  # type: ignore[misc]


def _to_jsonable(x: Any) -> Any:
    """Convert common RL objects (numpy scalars/arrays, dicts) into JSON-serializable forms."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # Fallback: at least preserve something printable
    return str(x)


def run_episode(
    agent: AgentLike,
    cfg: RunConfig = RunConfig(),
) -> RunResult:
    """
    Run a single episode.

    What gets saved (when cfg.out_dir != None):
      - metrics.json (episode summary + config)
      - run_info.jsonl (per-step JSON lines, including `obs`, best-effort `state`, and `info`) (if cfg.save_run_info)
      - frames/000000.png ... (if cfg.save_frames)
      - video.mp4 (if cfg.save_video)
    """

    # Decide render_mode:
    # - For frames/video we need rgb_array.
    # - For pure agent training/eval (no visuals), None is fastest.
    render_mode: Optional[str] = "rgb_array" if (cfg.save_frames or cfg.save_video) else None

    # Create output structure
    run_dir = _resolve_run_dir(cfg)
    frames_dir: Optional[Path] = None
    video_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    run_info_path: Optional[Path] = None

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        if cfg.save_run_info:
            run_info_path = run_dir / "run_info.jsonl"
        if cfg.save_frames:
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
        if cfg.save_video:
            video_path = run_dir / "video.mp4"

    # Create env
    env_kwargs: Dict[str, Any] = {
        "render_mode": render_mode,
    }
    if cfg.disable_env_checker:
        env_kwargs["disable_env_checker"] = True
    if cfg.use_lidar is not None:
        env_kwargs["use_lidar"] = bool(cfg.use_lidar)

    env = gym.make(cfg.env_id, **env_kwargs)

    t0 = time.time()
    obs, info = env.reset(seed=cfg.seed)
    processed_obs = _process_obs(obs)

    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    last_score: Optional[int] = info.get("score") if isinstance(info, dict) else None

    # If saving run info, stream per-step JSONL (avoid holding large info dicts in RAM).
    run_info_f = None
    if run_info_path is not None:
        run_info_f = run_info_path.open("w", encoding="utf-8", buffering=1)

        # record reset info
        run_info_f.write(
            json.dumps(
                {
                    "event": "reset",
                    "info": _to_jsonable(info),
                    "obs": _to_jsonable(obs),
                    "t_wall_s": 0.0,
                },
            )
            + "\n"
        )

    # If saving video, stream frames directly to writer (avoid holding all frames in RAM).
    video_writer = None
    if video_path is not None:
        # imageio v2 writer uses imageio-ffmpeg for mp4 on most setups
        video_writer = imageio.get_writer(str(video_path), fps=int(cfg.video_fps))

    try:
        step_idx = 0
        while True:
            action = _agent_action(agent, processed_obs)

            obs, reward, terminated, truncated, info = env.step(action)
            processed_obs = _process_obs(obs)
            # info is just dict with score

            r = float(reward)
            last_score = info.get("score") if isinstance(info, dict) else last_score

            if run_info_f is not None:
                run_info_f.write(
                    json.dumps(
                        {
                            "step": step_idx,
                            "t_wall_s": float(time.time() - t0),
                            "action": int(action),
                            "reward": float(r),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "score": int(last_score) if last_score is not None else None,
                            "processed_obs": _to_jsonable(processed_obs),
                            "info": _to_jsonable(info),
                        },
                    )
                    + "\n"
                )

            episode_return += r
            episode_length += 1

            if render_mode == "rgb_array" and (frames_dir is not None or video_writer is not None):
                frame = env.render()
                if frame is not None:
                    frame_np = np.asarray(frame)
                    if frames_dir is not None:
                        frame_path = frames_dir / f"{step_idx:06d}.png"
                        iio.imwrite(frame_path, frame_np)
                    if video_writer is not None:
                        video_writer.append_data(frame_np)

            step_idx += 1
            if terminated or truncated:
                break
            if cfg.max_steps is not None and episode_length >= cfg.max_steps:
                truncated = True
                break
    finally:
        if run_info_f is not None:
            try:
                run_info_f.close()
            except Exception:
                pass
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass
        env.close()

    wall_time_s = time.time() - t0

    result = RunResult(
        config=cfg, # 
        episode_return=float(episode_return), # 
        episode_length=int(episode_length),# 
        terminated=bool(terminated),# 
        truncated=bool(truncated),# 
        score=int(last_score) if last_score is not None else None,# 
        wall_time_s=float(wall_time_s),# 
        run_dir=str(run_dir) if run_dir is not None else None,# 
        frames_dir=str(frames_dir) if frames_dir is not None else None, # 
        video_path=str(video_path) if video_path is not None else None, # 
        metrics_path=str(metrics_path) if metrics_path is not None else None,# 
        run_info_path=str(run_info_path) if run_info_path is not None else None,# 
    )

    if run_dir is not None and metrics_path is not None:
        _save_run_outputs(result, Path(metrics_path))

    return result


def _save_run_outputs(result: RunResult, metrics_path: Path) -> None:
    # metrics.json
    metrics: Dict[str, Any] = {
        "config": asdict(result.config),
        "summary": {
            "episode_return": result.episode_return,
            "episode_length": result.episode_length,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "score": result.score,
            "wall_time_s": result.wall_time_s,
        },
        "output_directories": {
            "run_dir": result.run_dir,
            "frames_dir": result.frames_dir,
            "video_path": result.video_path,
            "metrics_path": result.metrics_path,
            "run_info_path": result.run_info_path,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")