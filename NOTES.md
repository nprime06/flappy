If you are claude or cursor, don't edit this file!

Personal notes for what I will need to do later on to clean up the codebase. 

Usage and file structure:
- load real game from game/rl/environment.py
- train ppo in game/rl/
- record gameplay (with augmented policy) in vod/run to vod/
- train VAE in diffuse/ae/
- encode latents (update LATENT_MEAN/STD!) to latent-vod/
- train flow and reflow in diffuse/ngen/

Refactor/cleanup 1/24: 
- cleaned up stupid ass claude code in nn/
- need to change dataloader to return k actions total including current action
- 

# VOD: the very first frame right after gym.make is NOT saved

# VOD: 0.0 is acutally 0.005, 0.2 FML

# there are a few hardcoded things, like observation normalization, bird detection 

initial latents: it is out of distribution to start from data -> data (need to refresh noise at inference ?) but still is interesting idea to test out? 

definitely want to compress all training data to latent before doing anything # DONE
- assume that latent-vod/ latents are already normalized
- vae_20260125_125114 accidentally used LATENT_MEAN = 0.4755, LATENT_STD = 1.5959 instead of correct LATENT_MEAN = 0.4735, LATENT_STD = 1.5931 but that should be okay

conditioning question. conceptually i think action conditioning is very important for ensuring consistent training dynamics, whereas the past frames conditioning kind of just tells you how the pipes arem oving (quite predictable and whatnot) idkdidiidkdk its prolly impractical to have different amount of frame and action conditioning?  # DONE

think carefully abt reflow, we still have to train with noise aug, etc. only diff is to change choosing x0~N(0,1)


Change vod/record.py to take input flags for consistency


Doing detailed ablation study for the optimizations we used? idk 