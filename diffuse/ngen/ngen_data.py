from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TraceDataset(Dataset):
    """Dataset that returns (past_k_frames, current_frame, action) for world model training.
    
    Supports wrap-around: when an episode starts, its first frames use the last k frames
    from the previous episode as context, treating the dataset as one continuous sequence.
    """

    def __init__(self, vod_dir, k=4):
        import json
        self.k = k
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Collect episodes in order (sorted by path for consistency)
        episodes = []
        for run_dir in sorted(Path(vod_dir).glob("*/*/")):
            frames_dir = run_dir / "frames"
            info_file = run_dir / "run_info.jsonl"
            if not frames_dir.exists() or not info_file.exists():
                continue

            # Parse actions from run_info.jsonl
            actions = {}
            with open(info_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if "step" in rec:
                        actions[rec["step"]] = rec["action"]

            n_frames = len(list(frames_dir.glob("*.png")))
            if n_frames > 0:
                episodes.append({
                    "frames_dir": frames_dir,
                    "n_frames": n_frames,
                    "actions": actions,
                })

        if len(episodes) == 0:
            raise ValueError(f"No valid episodes found in {vod_dir}")

        # Build index: list of (episode_idx, step_idx, action) for valid samples
        # With wrap-around, we can use every frame as a target (except first k of first episode)
        self.samples = []
        self.episodes = episodes
        
        for ep_idx, episode in enumerate(episodes):
            n_frames = episode["n_frames"]
            actions = episode["actions"]
            
            # First episode: start from step k (need k prior frames within episode)
            # Subsequent episodes: start from step 0 (can use previous episode's last k frames)
            start_step = k if ep_idx == 0 else 0
            
            for step in range(start_step, n_frames):
                if step in actions:
                    self.samples.append((ep_idx, step, actions[step]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, step, action = self.samples[idx]
        episode = self.episodes[ep_idx]
        frames_dir = episode["frames_dir"]
        n_frames = episode["n_frames"]

        # Load current frame
        current = Image.open(frames_dir / f"{step:06d}.png").convert("RGB")
        current = self.transform(current)

        # Load past k frames with wrap-around support
        past = []
        for offset in range(self.k):
            # Calculate which frame index we need: frames [step-k, step-k+1, ..., step-1]
            frame_idx = step - self.k + offset
            
            # If frame_idx < 0, we need to get it from the previous episode
            if frame_idx < 0:
                if ep_idx == 0:
                    # First episode: shouldn't happen if start_step=k, but handle gracefully
                    # Use frame 0 as fallback
                    frame_idx = 0
                    use_frames_dir = frames_dir
                else:
                    # Use last frames from previous episode
                    prev_episode = self.episodes[ep_idx - 1]
                    use_frames_dir = prev_episode["frames_dir"]
                    prev_n_frames = prev_episode["n_frames"]
                    # frame_idx is negative, so we want prev_n_frames + frame_idx
                    frame_idx = prev_n_frames + frame_idx
            else:
                use_frames_dir = frames_dir
            
            frame = Image.open(use_frames_dir / f"{frame_idx:06d}.png").convert("RGB")
            past.append(self.transform(frame))
        
        past = torch.stack(past)  # (k, C, H, W)

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