# Playability Eval

- NGEN: `diffuse/ngen/runs/ngen_20260207_005513/checkpoints/latest.pt`
- VAE: `diffuse/vae/runs/vae_20260201_111550/checkpoints/latest.pt`
- Device: `mps`
- Windows: 4
- Rollout length: 10
- Euler steps: 8

## One-Step

### random:cfg1
- target MAE: 0.0844
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2090
- action diff / real delta: 0.163
- noise std / real delta: 0.324
- clamp frac: 0.0141

### random:cfg1.5
- target MAE: 0.0840
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2093
- action diff / real delta: 0.166
- noise std / real delta: 0.321
- clamp frac: 0.0142

### zero:cfg1
- target MAE: 0.0477
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.1785
- action diff / real delta: 0.160
- noise std / real delta: 0.000
- clamp frac: 0.0143

### corr:cfg1
- target MAE: 0.0847
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2091
- action diff / real delta: 0.162
- noise std / real delta: 0.000
- clamp frac: 0.0142

## Rollout

### random:cfg1
- mean MAE: 0.0988
- final MAE: 0.1127
- MAE slope/frame: 0.00348
- generated delta / true delta: 0.953
- clamp frac: 0.0154

### random:cfg1.5
- mean MAE: 0.0982
- final MAE: 0.1096
- MAE slope/frame: 0.00309
- generated delta / true delta: 0.954
- clamp frac: 0.0153

### zero:cfg1
- mean MAE: 0.0650
- final MAE: 0.0778
- MAE slope/frame: 0.00346
- generated delta / true delta: 0.853
- clamp frac: 0.0155

### corr:cfg1
- mean MAE: 0.0992
- final MAE: 0.1122
- MAE slope/frame: 0.00335
- generated delta / true delta: 0.797
- clamp frac: 0.0152
