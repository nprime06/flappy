"""PPO agent for Flappy Bird."""

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from environment import run_episode, RunConfig

CONFIG = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs_per_update": 4,
    "batch_size": 16,
    "num_episodes": 100,
    "log_interval": 100,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "runs_dir": "/home/willzhao/flappy/game/rl/runs",
}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.actor = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        h = self.shared(obs)
        return self.actor(h).squeeze(-1), self.critic(h).squeeze(-1)


class PPOAgent:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.obs_list = []
        self.actions = []
        self.log_probs = []
        self.values = []

    def act(self, obs):
        self.obs_list.append(list(obs))
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob, value = self.model(obs_t)
            dist = torch.distributions.Bernoulli(prob)
            action = dist.sample()

        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action).item())
        self.values.append(value.item())
        return int(action.item())

    def reset(self):
        self.obs_list, self.actions, self.log_probs, self.values = [], [], [], []


def compute_gae(rewards, values, gamma, lam):
    advantages, gae = [], 0.0
    values = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def train(run_dir=None):
    device = torch.device(CONFIG["device"])
    model = ActorCritic(obs_dim=3).to(device)
    optimizer = Adam(model.parameters(), lr=CONFIG["lr"])

    # Setup run directory
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(CONFIG["runs_dir"], f"ppo_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "training_log.jsonl")
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")

    print(f"Run directory: {run_dir}")

    # Try to resume from checkpoint
    start_ep = 0
    running_score, running_length = 0.0, 0.0
    total_wall_time = 0.0

    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_ep = ckpt["episode"]
        running_score = ckpt["running_score"]
        running_length = ckpt["running_length"]
        total_wall_time = ckpt["wall_time_s"]
        print(f"Resumed at episode {start_ep}, avg_score={running_score:.1f}")

    all_obs, all_actions, all_log_probs, all_advantages, all_returns = [], [], [], [], []

    log_file = open(log_path, "a", buffering=1)
    train_start_t = time.perf_counter()

    for ep in range(start_ep, CONFIG["num_episodes"]):
        agent = PPOAgent(model, device)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RunConfig(out_dir=tmpdir, run_name="ep", save_run_info=True)
            result = run_episode(agent, cfg)

            rewards = []
            with open(result.run_info_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if "reward" in data:
                        rewards.append(data["reward"])

        advantages, returns = compute_gae(
            rewards, agent.values, CONFIG["gamma"], CONFIG["gae_lambda"]
        )

        all_obs.extend(agent.obs_list)
        all_actions.extend(agent.actions)
        all_log_probs.extend(agent.log_probs)
        all_advantages.extend(advantages)
        all_returns.extend(returns)

        alpha = 0.05
        running_score = alpha * (result.score or 0) + (1 - alpha) * running_score
        running_length = alpha * result.episode_length + (1 - alpha) * running_length

        if (ep + 1) % CONFIG["batch_size"] == 0:
            obs_t = torch.FloatTensor(all_obs).to(device)
            actions_t = torch.FloatTensor(all_actions).to(device)
            old_log_probs_t = torch.FloatTensor(all_log_probs).to(device)
            advantages_t = torch.FloatTensor(all_advantages).to(device)
            returns_t = torch.FloatTensor(all_returns).to(device)

            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            for _ in range(CONFIG["epochs_per_update"]):
                prob, values = model(obs_t)
                dist = torch.distributions.Bernoulli(prob)
                log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy()

                ratio = torch.exp(log_probs - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - CONFIG["clip_eps"], 1 + CONFIG["clip_eps"]) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values - returns_t) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + CONFIG["value_coef"] * value_loss + CONFIG["entropy_coef"] * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_obs, all_actions, all_log_probs, all_advantages, all_returns = [], [], [], [], []

            wall_time_s = total_wall_time + (time.perf_counter() - train_start_t)

            log_file.write(json.dumps({
                "episode": ep + 1,
                "avg_score": running_score,
                "avg_length": running_length,
                "wall_time_s": wall_time_s,
            }) + "\n")

        if (ep + 1) % CONFIG["log_interval"] == 0:
            wall_time_s = total_wall_time + (time.perf_counter() - train_start_t)

            print(
                f"Episode {ep+1:6d} | "
                f"score={result.score:2d} | "
                f"len={result.episode_length:3d} | "
                f"avg_score={running_score:.1f} | "
                f"avg_len={running_length:.0f}"
            )

            # Save checkpoint
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": ep + 1,
                "running_score": running_score,
                "running_length": running_length,
                "wall_time_s": wall_time_s,
            }
            torch.save(ckpt, latest_ckpt_path)
            torch.save(ckpt, os.path.join(checkpoint_dir, f"ep_{ep+1:07d}.pt"))

    log_file.close()
    print(f"\nTraining complete. Run directory: {run_dir}")


def evaluate(checkpoint_path, out_dir="eval_runs"):
    device = torch.device(CONFIG["device"])
    model = ActorCritic(obs_dim=3).to(device)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    agent = PPOAgent(model, device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory to resume from (creates new if not specified)")
    args = parser.parse_args()
    train(run_dir=args.run_dir)
