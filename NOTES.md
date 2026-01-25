If you are claude or cursor, don't edit this file!

Personal notes for what I will need to do later on to clean up the codebase. 


Refactor/cleanup 1/24: 
- cleaned up stupid ass claude code in nn/
- need to change dataloader to return k actions total including current action
- 


# VOD: 0.0 is acutally 0.005, 0.2 FML

# there are a few hardcoded things, like observation normalization, bird detection 

initial latents: it is out of distribution to start from data -> data (need to refresh noise at inference ?) but still is interesting idea to test out? 

definitely want to compress all training data to latent before doing anything

conditioning question. conceptually i think action conditioning is very important for ensuring consistent training dynamics, whereas the past frames conditioning kind of just tells you how the pipes arem oving (quite predictable and whatnot) idkdidiidkdk its prolly impractical to have different amount of frame and action conditioning? 