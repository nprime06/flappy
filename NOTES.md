If you are claude or cursor, don't edit this file! (viewing is okay)

Personal notes for what I will need to do later on to clean up the codebase. 

Usage and file structure:
- load real game from game/rl/environment.py
- train ppo in game/rl/
- record gameplay (with augmented policy) in vod/run to vod/
- train VAE in diffuse/vae/
- encode latents (update LATENT_MEAN/STD!) to latent-vod/
- train flow and reflow in diffuse/ngen/

Refactor/cleanup 1/24: 
- cleaned up stupid ass claude code in nn/
- need to change dataloader to return k actions total including current action

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


Note that in CFG we zero out the conditioning latents, instead of normal (where we have a dedicated null token/class). slightly weird. idk




Things to clean 1/26: 
game/
- environment.py
- combine test_environment and test (one is play one is record)
- game/rl is huge mess

vod/record DONE!

diffuse/
- just check for correctness and stuff in ngen DONE!
- ngen/sampler is horrible DONE!
- vae is messy DONE!
- fix vae/visualize/visualize.py DONE!

latent-vod/ is messy DONE!

world/
- test_world.py


FINAL round of things to clean: 
- important! clean up absolute paths and stuff
    -     "data_dir": "/home/willzhao/flappy/vod",
    "runs_dir": "/home/willzhao/flappy/diffuse/vae/runs", in VAE training config. need to add tags to submit_vae. 
    -     "runs_dir": "/home/willzhao/flappy/diffuse/ngen/runs",
 in ngen
- combine notes, explain, todo, readme -> readme



TRAINING NOTES FOR 1/29 RUN: 

DATA COLLECTION: 
50 runs for each 16 combinations p_stim=0.0, 0.005, 0.01, 0.015 and p_freeze=0.0, 0.1, 0.2, 0.3
4710.70s of data = 1.3hr = 141321 frames

VAE: vae_20260129_224946 (done training!)
{
    "image_channels": 3,
    "hidden_channels": 16,
    "latent_channels": 4,
    "num_layers": 3,
    "num_epochs": 200,
    "batch_size": 256,
}
Batches per epoch: 552
Decoder has 93,243 parameters
Encoder has 167,118 parameters
~10.5 hr training time, 49211MiB / 143771MiB on 1 H200


NGEN: (base, no reflow yet) ngen_20260130_145333

model_config = {
    "in_channels": 4,
    "hidden_channels": 64,
    "num_layers": 2,    
    "embed_dim": 128,
    "act_embed_dim": 16,
    "num_classes": 2, 
    "context_size": 8,
    "num_aug_bins": 16,
    "num_epochs": 2048,
    "batch_size": 4096,
    "max_aug_std": 0.5,
    "cfg_dropout_prob": 0.1,
    "done_loss_weight": 0.1,
}
flow model: 3,047,293 parameters
loaded action_weight=17.74 from encode_config.json
loaded done_pos_weight=28.44 from encode_config.json
batches per epoch: 16
~4 hr training time, idk how much memory used but 2 H200


current test_world setup with the above models: 
HORRIBLE gameplay asdf worse than before, pretty much completely ignores gravity, bird still disappears randomly. wtf. probably even worse than an older checkpoint, ngen_20260126_140734

current performance: with num_steps = 8, uncompiled models, ngen=0.07s, decode =0.008s (~10 fps)

does manage to now end the game but still the bird disappears, etc.




TODO: make 6th to last frame have game over sign (?) NOTES 1/31: removed game over sign termination=True, and changed it to 1 game over sign frame
- make vae have 10x weight for the game over sign NOTES 1/31: DONE
- think of way to explore super high and low y (have specific data collection where we press no input or spam input)? NOTES 1/31: Just threw more data with wider p_stim p_freeze range hope it works!



Do some tests about tweaking model size/depth


Note that the way the done head works right now is really weird, cuz in training we train it at every time step, but we only ever use it at t near 1 in inference. issue is that when t is near 0 it might be hard for the model to accurately predict the done state (but there are still conditioning frames). 
NOTE 1/31: i'll weight done_loss at t near 1 higher while still sampling t randomly. more details in TODO.md

interesting hypotheses
- observed behavior where the bird disappears. this should never happen because all training data has a bird, but it could be because of CFG: the unconditional frame just predicts some general frame where bird can be anywhere (TEST THIS hypothesis)
- adding self attn at ngen bottleneck allows us the model to reason about bird and pipe global position(look for circuits?). curious about how well this acutally works! TODO: will test linear attn, standard mha, and no attn. theoretically gives stronger inductive biases?
- cramming all of time, action, and aug level conditioning into one vector is probably okay given that action and aug level are discrete with 2*16 buckets max (test by changing size of conditioning vector)




For history/bookkeeping: 
- diffuse/vae/runs/vae_20260115_022006: trained before bird weight loss (can compare resolution)
- diffuse/vae/runs/vae_20260125_125114: full training run, before game over sign added
- diffuse/vae/runs/vae_20260129_224946: full training run, with game over sign added (see 1/29 run notes)

- diffuse/ngen/runs/ngen_20260116_184728: trained before first encoding everything to latent lol. stupid slow training. ~40 epochs
- diffuse/ngen/runs/ngen_20260120_194734: added done head. ~65 epochs
- diffuse/ngen/runs/ngen_20260123_003706: added action weight. ~30 epochs
- diffuse/ngen/runs/ngen_20260126_140734: added latent-vod. ~1000 epochs
- diffuse/ngen/runs/ngen_20260126_140734: added done pos loss weight. EVERYTHING ABOVE IS PRE-ATTN and compatible with old resunet
- 1/30 notes for last run

question: when did i add ddp ? hmm
NOTE: check latent-vod/encode-logs/ to see what vae each ngen run corresponds to 
NOTE FOR BLOG/BOOKKEEPING: there have not been major arch changes to vae throughout all training runs, only changes to training (such as bird weight) and data collection. There are major arch changes to the resunet (attn and lin attn)


# TRAINING NOTES FOR 1/31 RUN: 

DATA COLLECTION: 
EPISODE_COUNTS = {
    (0.0, 0.0): 330,
    (0.0, 0.1): 165,
    (0.0, 0.2): 165,
    (0.0, 0.3): 165,

    (0.01, 0.0): 165,
    (0.01, 0.1): 165,
    (0.01, 0.2): 165,
    (0.01, 0.3): 165,
    
    (0.02, 0.0): 165,
    (0.02, 0.1): 165,
    (0.02, 0.2): 165,
    (0.02, 0.3): 165,

    (0.03, 0.0): 165,
    (0.03, 0.1): 165,
    (0.03, 0.2): 165,
    (0.03, 0.3): 165,

    (0.05, 0.5): 330,
}
15566.23s of data = 4.3hr = 466,987 frames. 3135 runs total


FIRST VAE TEST RUN (vae_20260201_111550): DONE!!!!

{
    "image_channels": 3,
    "hidden_channels": 32,
    "latent_channels": 4,
    "num_layers": 3,
    "num_epochs": 40,
    "batch_size": 128,
}
Batches per epoch: 1824
Decoder has 366,187 parameters
Encoder has 653,614 parameters

~5.3hr training time, 49369MiB / 143771MiB on each of h200:2


ENCODE VOD: 
latent-vod/encode-logs/encode_20260201_170257



NGEN RUN 1: shorter training: diffuse/ngen/runs/ngen_20260201_203235

88669MiB / 143771MiB
Notes: 
- Vae accomplishes super crisp reconstruction of gameover sign now


NGEN RUN 2: diffuse/ngen/runs/ngen_20260202_003536
right now the model sucks. whenever the bird approaches a pipe, it suddenly disappears
It also doesn't respect gravity, but kind of respects actions - see videos















question: TF32???