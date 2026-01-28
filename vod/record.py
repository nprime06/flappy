import os
import random
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.environment import run_episode, RunConfig
from game.rl.train_ppo import ActorCritic, PPOAgent
from tqdm import tqdm

PPO_RUN_ID = "20260108_034928"
CHECKPOINT = "latest.pt"

episode_counts = {
    (0.0, 0.0): 5,
    (0.1, 0.1): 5,
}

RUNS_DIR = "/Users/william/Desktop/Random/flappy/game/rl/runs"
VOD_BASE_DIR = "/Users/william/Desktop/Random/flappy/vod"

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

        a = int(action.item())
        if a == 0:
            if random.random() < self.p_stim:
                a = 1
        else:
            if random.random() < self.p_freeze:
                a = 0

        return a


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(RUNS_DIR, f"ppo_{PPO_RUN_ID}", "checkpoints", CHECKPOINT)
    print(f"Loading checkpoint: {checkpoint_path}")

    model = ActorCritic(obs_dim=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for (p_stim, p_freeze), num_episodes in episode_counts.items():
        print(f"\n=== Recording {num_episodes} episodes with p_stim={p_stim}, p_freeze={p_freeze} ===")

        agent = HijackedPPOAgent(model, device, p_stim=p_stim, p_freeze=p_freeze)
        output_dir = os.path.join(VOD_BASE_DIR, f"p_stim_{p_stim}_p_freeze_{p_freeze}")

        cfg = RunConfig(
            out_dir=output_dir,
            save_run_info=True,
            save_frames=True,
            save_video=False,
        )

        total_length = 0
        for ep in range(num_episodes):
            result = run_episode(agent, cfg)
            total_length += result.episode_length
            print(f"  Episode {ep + 1}/{num_episodes}; length: {result.episode_length/30:.2f}s; total: {total_length/30:.2f}s")

if __name__ == "__main__":
    main()
