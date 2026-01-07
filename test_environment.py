#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

from environment import RunConfig, run_episode


def main() -> None:
    p_flap = 0.1
    save_video = False
    out_dir = "test_runs"
    save_run_info = True
    save_frames = True
    save_video = True

    def agent(_obs) -> int:
        # FlappyBird action space is effectively Discrete(2): 0=no-op, 1=flap.
        return 1 if (random.random() < p_flap) else 0


    run_config = RunConfig(
        env_id="FlappyBird-v0", 

        out_dir=out_dir,
        save_run_info=save_run_info,
        save_frames=save_frames,
        save_video=save_video,
    )

    run_episode(agent, run_config)


if __name__ == "__main__":
    main()




# def run_episode(
#     agent: AgentLike,
#     cfg: RunConfig = RunConfig(),

# @dataclass(frozen=True)
# class RunConfig:
#     env_id: str = "FlappyBird-v0"
#     seed: Optional[int] = None
#     max_steps: Optional[int] = None

#     # Performance knobs
#     use_lidar: bool = False
#     disable_env_checker: bool = True

#     # Saving
#     out_dir: Optional[str] = "runs"  # set to None to disable any file output
#     run_name: Optional[str] = None  # default is timestamp
#     # If True, stream per-timestep info to disk as the run progresses (JSONL).
#     # This is useful for long runs where you don't want to lose info if interrupted.
#     save_run_info: bool = False
#     save_frames: bool = False
#     save_video: bool = False
#     video_fps: int = 30



