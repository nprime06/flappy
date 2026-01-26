"""Dataset for pre-computed VAE latents.

This module provides LatentTraceDataset which has the same interface as
TraceDataset but loads pre-computed latents instead of raw images.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class LatentTraceDataset(Dataset):
    """
    past_latents: (k, 4, H', W')
    current_latent: (4, H', W')
    all_actions: (k+1,) of actions a_{n-k}, ..., a_{n}; note a_n -> frame_n
    done: 0, 1
    """

    def __init__(self, latent_vod_dir, k=4):
        self.k = k

        # Build index: list of (run_path, step_idx, action, done) for valid samples
        self.samples = []
        action_0_samples = []
        action_1_samples = []

        # Store actions dict per episode for looking up past actions
        self.episode_actions = {}

        # Cache loaded latents per run (lazy loading)
        self._latent_cache = {}

        for run_dir in Path(latent_vod_dir).glob("*/*/"):
            latents_file = run_dir / "latents.pt"
            info_file = run_dir / "run_info.jsonl"
            if not latents_file.exists() or not info_file.exists():
                continue

            # Parse actions AND done flags from run_info.jsonl
            actions = {}
            done_flags = {}
            with open(info_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if "step" in rec:
                        actions[rec["step"]] = rec["action"]
                        done_flags[rec["step"]] = rec.get("terminated", False) or rec.get("truncated", False)

            # Store actions dict for this episode (keyed by run_dir path)
            run_key = str(run_dir)
            self.episode_actions[run_key] = actions

            # Get number of frames from latents file
            latents = torch.load(latents_file, map_location="cpu", weights_only=True)
            n_frames = latents.shape[0]

            # Cache latents
            self._latent_cache[run_key] = latents

            # Valid samples: steps where we have k prior frames within the same episode
            for step in range(k, n_frames):
                if step in actions:
                    done = done_flags.get(step, step == n_frames - 1)
                    sample = (run_key, step, actions[step], done)
                    self.samples.append(sample)

                    if actions[step] == 0:
                        action_0_samples.append(sample)
                    else:
                        action_1_samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        run_key, step, _action, done = self.samples[idx]  # _action unused, we build all k+1 from dict

        # Get cached latents for this run
        latents = self._latent_cache[run_key]

        # Get current latent
        current_latent = latents[step]  # (4, H', W')

        # Get past k latents
        past_latents = latents[step - self.k:step]  # (k, 4, H', W')

        # Load k+1 actions: [step-k, step-k+1, ..., step] inclusive
        # These are the k actions that caused the k context latents, plus the action that causes the target latent
        actions_dict = self.episode_actions[run_key]
        actions = []
        for i in range(step - self.k, step + 1):  # step-k to step inclusive = k+1 actions
            actions.append(actions_dict.get(i, 0))
        actions = torch.tensor(actions, dtype=torch.long)  # (k+1,)

        return past_latents, current_latent, actions, torch.tensor(done, dtype=torch.float)


if __name__ == "__main__":
    # Test with local latent-vod directory
    dataset = LatentTraceDataset("/Users/william/Desktop/Random/flappy/latent-vod")
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        past_latents, current_latent, actions, done = next(iter(loader))
        print(f"Past latents shape: {past_latents.shape}")
        print(f"Current latent shape: {current_latent.shape}")
        print(f"Actions shape: {actions.shape}")  # Should be (B, k+1) = (32, 5) for k=4
        print(f"Done shape: {done.shape}")
        print(f"Past latents range: [{past_latents.min():.2f}, {past_latents.max():.2f}]")
        print(f"Current latent range: [{current_latent.min():.2f}, {current_latent.max():.2f}]")
