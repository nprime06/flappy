## project details

game: flappy environment

- flappy environment, takes in a decision-making agent and outputs gameplay
- train_ppo creates decent bot which averages ~5 score

vod: 

- gets video rollouts from ppo bot; to better explore game space we hijack bot's decisions with some p
- latent t-1, ... + action t -> latent t

diffuse: model and training

- gamengen-inspired: low-sampling-step direct conditioning (through reflows) on previous latents

world: deployment




## File Structure

```
flappy/
├── game/
│   ├── environment.py
│   ├── rl/
│   │   ├── train_ppo.py
│   │   ├── test_ppo.py
│   │   └── runs/ 
│
├── vod/                      # Recorded gameplay data
│   ├── record.py               # Script to batch-record episodes
│   ├── raw/                    # Raw recordings from PPO agent
│   │   └── {episode_id}/
│   │       ├── frames/         # PNG frames
│   │       ├── video.mp4
│   │       └── run_info.jsonl  # Actions, rewards, observations
│   ├── processed/              # Preprocessed for training
│   │   ├── frames/             # Resized/normalized frames
│   │   └── metadata.parquet    # Actions + frame paths
│   └── README.md
│
├── diffuse/                    # Diffusion model training
│   ├── model/
│   │   ├── unet.py             # U-Net architecture
│   │   ├── vae.py              # VAE encoder/decoder (or use pretrained)
│   │   └── diffusion.py        # Diffusion/reflow logic
│   ├── data/
│   │   ├── dataset.py          # PyTorch dataset for frame sequences
│   │   └── transforms.py       # Augmentations, normalization
│   ├── train.py                # Training loop
│   ├── config.py               # Hyperparameters
│   ├── runs/                   # Training runs, checkpoints, logs
│   └── README.md
│
├── world/                    # World model deployment
│   ├── inference.py            # Run world model interactively
│   ├── server.py               # Optional: serve as API
│   ├── play.py                 # Playable demo with keyboard input
│   └── README.md
│
├── scripts/                    # Cross-cutting utilities
│   ├── visualize.py            # Compare real vs generated
│   └── metrics.py              # FVD, LPIPS, etc.
│
└── README.md
```
