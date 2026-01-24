"""Dataset for pre-computed VAE latents.

This module provides LatentTraceDataset which has the same interface as
TraceDataset but loads pre-computed latents instead of raw images.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class LatentTraceDataset(Dataset):
    """Dataset that returns pre-computed latents for world model training.

    Same interface as TraceDataset but loads pre-computed latents from disk.
    Latents are already normalized (normalization baked in during encoding).

    Returns:
        past_latents: (k, 4, H', W') tensor of past k latents (already normalized)
        current_latent: (4, H', W') tensor of current latent (already normalized)
        past_actions: (k,) tensor of past k actions
        action: scalar tensor of current action
        done: (optional) scalar tensor indicating episode termination
    """

    def __init__(self, latent_vod_dir, k=4, include_done=False, balance_actions=False):
        self.k = k
        self.include_done = include_done

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

        # Balance actions by oversampling minority class
        if balance_actions and len(action_0_samples) > 0 and len(action_1_samples) > 0:
            n_action_0 = len(action_0_samples)
            n_action_1 = len(action_1_samples)
            print(f"Before balancing: action_0={n_action_0}, action_1={n_action_1}")

            if n_action_1 < n_action_0:
                oversample_factor = n_action_0 // n_action_1
                extra_samples = action_1_samples * (oversample_factor - 1)
                self.samples.extend(extra_samples)
                print(f"After balancing: {len(self.samples)} total samples "
                      f"(added {len(extra_samples)} action=1 samples, {oversample_factor}x oversample)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        run_key, step, action, done = self.samples[idx]

        # Get cached latents for this run
        latents = self._latent_cache[run_key]

        # Get current latent
        current_latent = latents[step]  # (4, H', W')

        # Get past k latents
        past_latents = latents[step - self.k:step]  # (k, 4, H', W')

        # Load past k actions
        actions_dict = self.episode_actions[run_key]
        past_actions = []
        for i in range(step - self.k, step):
            past_actions.append(actions_dict.get(i, 0))
        past_actions = torch.tensor(past_actions, dtype=torch.long)

        if self.include_done:
            return past_latents, current_latent, past_actions, torch.tensor(action, dtype=torch.long), torch.tensor(done, dtype=torch.float)
        return past_latents, current_latent, past_actions, torch.tensor(action, dtype=torch.long)


if __name__ == "__main__":
    # Test with local latent-vod directory
    dataset = LatentTraceDataset("/Users/william/Desktop/Random/flappy/latent-vod")
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        past_latents, current_latent, past_actions, action = next(iter(loader))
        print(f"Past latents shape: {past_latents.shape}")
        print(f"Current latent shape: {current_latent.shape}")
        print(f"Past actions shape: {past_actions.shape}")
        print(f"Action shape: {action.shape}")
        print(f"Past latents range: [{past_latents.min():.2f}, {past_latents.max():.2f}]")
        print(f"Current latent range: [{current_latent.min():.2f}, {current_latent.max():.2f}]")
