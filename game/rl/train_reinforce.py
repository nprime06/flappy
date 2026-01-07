"""REINFORCE agent for Flappy Bird."""

import json
import tempfile
import torch
import torch.nn as nn
from torch.optim import Adam
from environment import run_episode, RunConfig
from tqdm import tqdm
CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "batch_size": 16,
    "num_episodes": 1000000,
    "checkpoint_path": "reinforce_flappy.pt",
    "log_interval": 10000,
}


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(12, 64), nn.GELU(), 
                    nn.Linear(64, 128), nn.GELU(), 
                    nn.Linear(128, 64), nn.GELU(), 
                    nn.Linear(64, 32), nn.GELU(), 
                    nn.Linear(32, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class ReinforceAgent:
    def __init__(self, policy):
        self.policy = policy
        self.log_probs = []

    def act(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        prob = self.policy(obs_t)
        dist = torch.distributions.Bernoulli(prob)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return int(action.item())

    def reset(self):
        self.log_probs = []


def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train():
    policy = Policy()
    optimizer = Adam(policy.parameters(), lr=CONFIG["lr"])
    optimizer.zero_grad()

    running_score = 0.0
    running_length = 0.0

    for ep in tqdm(range(CONFIG["num_episodes"])):
        agent = ReinforceAgent(policy)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RunConfig(out_dir=tmpdir, run_name="ep", save_run_info=True)
            result = run_episode(agent, cfg)

            rewards = []
            with open(result.run_info_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if "reward" in data:
                        rewards.append(data["reward"])

        returns = compute_returns(rewards, CONFIG["gamma"])
        log_probs = torch.cat(agent.log_probs)

        loss = -(log_probs * returns).sum() / CONFIG["batch_size"]
        loss.backward()

        if (ep + 1) % CONFIG["batch_size"] == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Tracking
        alpha = 0.05
        running_score = alpha * (result.score or 0) + (1 - alpha) * running_score
        running_length = alpha * result.episode_length + (1 - alpha) * running_length

        if (ep + 1) % CONFIG["log_interval"] == 0:
            print(
                f"Episode {ep+1:4d} | "
                f"score={result.score:2d} | "
                f"len={result.episode_length:3d} | "
                f"avg_score={running_score:.1f} | "
                f"avg_len={running_length:.0f}"
            )

    torch.save(policy.state_dict(), CONFIG["checkpoint_path"])
    print(f"\nSaved final checkpoint to {CONFIG['checkpoint_path']}")


def evaluate(checkpoint_path=CONFIG["checkpoint_path"], out_dir="eval_runs"):
    policy = Policy()
    policy.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    policy.eval()

    agent = ReinforceAgent(policy)
    cfg = RunConfig(
        out_dir=out_dir,
        save_run_info=True,
        save_frames=True,
        save_video=True,
    )
    result = run_episode(agent, cfg)
    print(f"Score: {result.score}, Length: {result.episode_length}")
    print(f"Video: {result.video_path}")
    return result


if __name__ == "__main__":
    train()
    #evaluate()