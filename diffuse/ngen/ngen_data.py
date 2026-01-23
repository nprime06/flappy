from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TraceDataset(Dataset):
    """Dataset that returns (past_k_frames, current_frame, action, [done]) for world model training.

    Each episode is independent: samples start from step k (k+1th frame) and go to the end.
    No mixing between episodes.
    """

    def __init__(self, vod_dir, k=4, include_done=False, balance_actions=False):
        import json
        self.k = k
        self.include_done = include_done
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Build index: list of (frames_dir, step_idx, action, done) for valid samples
        self.samples = []
        action_0_samples = []
        action_1_samples = []

        for run_dir in Path(vod_dir).glob("*/*/"):
            frames_dir = run_dir / "frames"
            info_file = run_dir / "run_info.jsonl"
            if not frames_dir.exists() or not info_file.exists():
                continue

            # Parse actions AND done flags from run_info.jsonl
            actions = {}
            done_flags = {}
            with open(info_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if "step" in rec:
                        actions[rec["step"]] = rec["action"]
                        # Check for termination or truncation flags
                        done_flags[rec["step"]] = rec.get("terminated", False) or rec.get("truncated", False)

            # Valid samples: steps where we have k prior frames within the same episode
            n_frames = len(list(frames_dir.glob("*.png")))
            for step in range(k, n_frames):
                if step in actions:
                    # Fallback: last frame is done if not explicitly marked
                    done = done_flags.get(step, step == n_frames - 1)
                    sample = (frames_dir, step, actions[step], done)
                    self.samples.append(sample)

                    # Track by action for balancing
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
                # Oversample action=1 to match action=0
                oversample_factor = n_action_0 // n_action_1
                extra_samples = action_1_samples * (oversample_factor - 1)
                self.samples.extend(extra_samples)
                print(f"After balancing: {len(self.samples)} total samples "
                      f"(added {len(extra_samples)} action=1 samples, {oversample_factor}x oversample)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_dir, step, action, done = self.samples[idx]

        # Load current frame
        current = Image.open(frames_dir / f"{step:06d}.png").convert("RGB")
        current = self.transform(current)

        # Load past k frames (all from the same episode)
        past = []
        for i in range(step - self.k, step):
            frame = Image.open(frames_dir / f"{i:06d}.png").convert("RGB")
            past.append(self.transform(frame))
        past = torch.stack(past)  # (k, C, H, W)

        if self.include_done:
            return past, current, torch.tensor(action, dtype=torch.long), torch.tensor(done, dtype=torch.float)
        return past, current, torch.tensor(action, dtype=torch.long)


if __name__ == "__main__":
    dataset = TraceDataset("/Users/william/Desktop/Random/flappy/vod")
    print(f"Dataset size: {len(dataset)}")
    past, current, action = next(iter(DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)))
    print(f"Past shape: {past.shape}")
    print(f"Current shape: {current.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Value range: [{past.min():.2f}, {past.max():.2f}]")
    print(f"Value range: [{current.min():.2f}, {current.max():.2f}]")
    print(f"Value range: [{action.min():.2f}, {action.max():.2f}]")

    print(f"Values: ", action)