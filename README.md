# project details

game: flappy environment

- flappy environment, takes in a decision-making agent and outputs gameplay
- train_ppo creates decent bot which averages ~5 score
- vod: gets video rollouts from ppo bot; to better explore game space we hijack bot's decisions with some p

video: hosts diffusion model and training

- gamengen-inspired: low-sampling-step direct conditioning (through reflows) on previous latents

world: deployment
