from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TraceDataset(Dataset):
    """Dataset that returns (past_k_frames, current_frame, action) for world model training.
    
    Each episode is independent: samples start from step k (k+1th frame) and go to the end.
    No mixing between episodes.
    """

    def __init__(self, vod_dir, k=4):
        import json
        self.k = k
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Build index: list of (frames_dir, step_idx, action) for valid samples
        self.samples = []
        for run_dir in Path(vod_dir).glob("*/*/"):
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

            # Valid samples: steps where we have k prior frames within the same episode
            n_frames = len(list(frames_dir.glob("*.png")))
            for step in range(k, n_frames):
                if step in actions:
                    self.samples.append((frames_dir, step, actions[step]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_dir, step, action = self.samples[idx]

        # Load current frame
        current = Image.open(frames_dir / f"{step:06d}.png").convert("RGB")
        current = self.transform(current)

        # Load past k frames (all from the same episode)
        past = []
        for i in range(step - self.k, step):
            frame = Image.open(frames_dir / f"{i:06d}.png").convert("RGB")
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