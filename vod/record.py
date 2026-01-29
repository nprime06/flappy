import os
import random
import sys
import argparse
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.environment import run_episode, RunConfig
from game.rl.train_ppo import ActorCritic, PPOAgent
from tqdm import tqdm

EPISODE_COUNTS = {
    (0.0, 0.0): 50,
    (0.0, 0.1): 50,
    (0.0, 0.2): 50,
    (0.0, 0.3): 50,

    (0.005, 0.0): 50,
    (0.005, 0.1): 50,
    (0.005, 0.2): 50,
    (0.005, 0.3): 50,

    (0.01, 0.0): 50,
    (0.01, 0.1): 50,
    (0.01, 0.2): 50,
    (0.01, 0.3): 50,

    (0.015, 0.0): 50,
    (0.015, 0.1): 50,
    (0.015, 0.2): 50,
    (0.015, 0.3): 50,
}

class HijackedPPOAgent(PPOAgent):
    def __init__(self, model, device, p_stim: float, p_freeze: float):
        super().__init__(model, device)
        self.p_stim = float(p_stim)
        self.p_freeze = float(p_freeze)

    def act(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prob, _value = self.model(obs_t)
            dist = torch.distributions.Bernoulli(prob)
            action = dist.sample()  # tensor scalar {0,1}

        a = int(action.item()) # hijack
        if a == 0:
            if random.random() < self.p_stim: a = 1
        else:
            if random.random() < self.p_freeze: a = 0

        return a


def main(checkpoint_path, vod_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint_path}")

    model = ActorCritic(obs_dim=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_length = 0

    for (p_stim, p_freeze), num_episodes in EPISODE_COUNTS.items():
        print(f"\n=== Recording {num_episodes} episodes with p_stim={p_stim}, p_freeze={p_freeze} ===")

        agent = HijackedPPOAgent(model, device, p_stim=p_stim, p_freeze=p_freeze)
        output_dir = os.path.join(vod_dir, f"p_stim_{p_stim}_p_freeze_{p_freeze}")

        cfg = RunConfig(
            out_dir=output_dir,
            save_run_info=True,
            save_frames=True,
            save_video=False,
        )

        specific_total_length = 0
        for ep in range(num_episodes):
            result = run_episode(agent, cfg)
            total_length += result.episode_length
            specific_total_length += result.episode_length
            print(f"  Ep {ep + 1}/{num_episodes}; len: {result.episode_length/30:.2f}s; tot: {total_length/30:.2f}s; avg: {specific_total_length/((ep + 1) * 30):.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    vod_dir = str(Path(__file__).parent)
    
    main(args.checkpoint, vod_dir)
