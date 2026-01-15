import os
import sys
import random
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import run_episode, RunConfig
from rl.train_ppo import ActorCritic, PPOAgent
from tqdm import tqdm

RUN_ID = "20260108_034928"
CHECKPOINT = "latest.pt"
# uses trained PPO checkpoint

RUNS_DIR = "/Users/william/Desktop/Random/flappy/game/rl/runs"
OUTPUT_DIR = "/Users/william/Desktop/Random/flappy/game/rl/trained_model"


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

        # print(f"prob={prob.item():.4f}, action={a}")
        return a


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(RUNS_DIR, f"ppo_{RUN_ID}", "checkpoints", CHECKPOINT)
    print(f"Loading checkpoint: {checkpoint_path}")

    model = ActorCritic(obs_dim=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    agent = HijackedPPOAgent(model, device, p_stim=0.005, p_freeze=0.2)

    cfg = RunConfig(
        out_dir=OUTPUT_DIR,
        save_run_info=True,
        save_frames=True,
        save_video=True,
    )

    print(f"Running evaluation...")
    for _ in tqdm(range(5)): 
        result = run_episode(agent, cfg)



if __name__ == "__main__":
    main()
