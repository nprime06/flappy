"""PPO agent for Flappy Bird."""

import json
import tempfile
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
    "num_episodes": 100000,
    "checkpoint_path": "ppo_flappy.pt",
    "log_interval": 100,
    "log_path": "ppo_training_log.jsonl",
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
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


def train():
    device = torch.device(CONFIG["device"])
    model = ActorCritic(obs_dim=3).to(device)
    optimizer = Adam(model.parameters(), lr=CONFIG["lr"])

    running_score, running_length = 0.0, 0.0
    all_obs, all_actions, all_log_probs, all_advantages, all_returns = [], [], [], [], []

    log_file = open(CONFIG["log_path"], "w", buffering=1)

    for ep in range(CONFIG["num_episodes"]):
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

        log_file.write(json.dumps({
            "episode": ep + 1,
            "score": result.score,
            "length": result.episode_length,
            "return": result.episode_return,
            "avg_score": running_score,
            "avg_length": running_length,
        }) + "\n")

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

        if (ep + 1) % CONFIG["log_interval"] == 0:
            print(
                f"Episode {ep+1:4d} | "
                f"score={result.score:2d} | "
                f"len={result.episode_length:3d} | "
                f"avg_score={running_score:.1f} | "
                f"avg_len={running_length:.0f}"
            )

    log_file.close()
    torch.save(model.state_dict(), CONFIG["checkpoint_path"])
    print(f"\nSaved final checkpoint to {CONFIG['checkpoint_path']}")
    print(f"Training log saved to {CONFIG['log_path']}")


def evaluate(checkpoint_path=CONFIG["checkpoint_path"], out_dir="eval_runs"):
    device = torch.device(CONFIG["device"])
    model = ActorCritic(obs_dim=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
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
    train()
