import os
import random
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.environment import run_episode, RunConfig
from game.rl.train_ppo import ActorCritic, PPOAgent
from tqdm import tqdm

RUN_ID = "20260108_034928"
CHECKPOINT = "latest.pt"
# uses trained PPO checkpoint

p_stim = 0.1
p_freeze = 0.1
num_episodes = 5

 

RUNS_DIR = "/Users/william/Desktop/Random/flappy/game/rl/runs"
OUTPUT_DIR = f"/Users/william/Desktop/Random/flappy/vod/p_stim_{p_stim}_p_freeze_{p_freeze}"


class HijackedPPOAgent(PPOAgent):
    def __init__(self, model, device, p_stim: float, p_freeze: float):
        super().__init__(model, device)
        self.p_stim = float(p_stim)
        self.p_freeze = float(p_freeze)

    def act(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
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

    checkpoint_path = os.path.join(RUNS_DIR, f"ppo_{RUN_ID}", "checkpoints", CHECKPOINT)
    print(f"Loading checkpoint: {checkpoint_path}")

    model = ActorCritic(obs_dim=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    agent = HijackedPPOAgent(model, device, p_stim=p_stim, p_freeze=p_freeze)

    cfg = RunConfig(
        out_dir=OUTPUT_DIR,
        save_run_info=True,
        save_frames=True,
        save_video=False,
    )

    total_length = 0
    for step in range(num_episodes): 
        result = run_episode(agent, cfg)
        total_length += result.episode_length
        print(f"Step {step + 1}/{num_episodes}; Episode length: {result.episode_length/30:.2f}s; total length: {total_length/30:.2f}s")

if __name__ == "__main__":
    main()
