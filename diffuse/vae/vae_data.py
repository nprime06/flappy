from collections import defaultdict
from pathlib import Path
from PIL import Image
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FramesDataset(Dataset): # sample random individual frames from all vod
    def __init__(self, vod_dir):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ]) # assumes images are in [0, 1]
        self.frame_paths = sorted(Path(vod_dir).glob("*/*/frames/*.png"))

        # mark last frame of each episode as game-over
        episodes = defaultdict(list)
        for i, path in enumerate(self.frame_paths):
            episodes[path.parent].append(i)
        last_frame_indices = set()
        for indices in episodes.values():
            last_frame_indices.add(indices[-1])
        self.is_gameover = [i in last_frame_indices for i in range(len(self.frame_paths))]

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        return self.transform(img), self.is_gameover[idx]


if __name__ == "__main__": # test
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-dir", type=str, required=True)
    args = parser.parse_args()

    dataset = FramesDataset(args.vod_dir)
    gameover_count = sum(dataset.is_gameover)
    print(f"total frames: {len(dataset)}, gameover frames: {gameover_count}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    batch, is_gameover = next(iter(loader))
    print(f"batch shape: {batch.shape}")  # [4, 3, 512, 288]
    print(f"value range: [{batch.min():.2f}, {batch.max():.2f}]")
    print(f"is_gameover: {is_gameover}")
