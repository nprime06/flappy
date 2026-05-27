# Playability Eval

- NGEN: `diffuse/ngen/runs/ngen_20260207_005513/checkpoints/latest.pt`
- VAE: `diffuse/vae/runs/vae_20260201_111550/checkpoints/latest.pt`
- Device: `mps`
- Windows: 4
- Rollout length: 10
- Euler steps: 20

## One-Step

### random:cfg1
- target MAE: 0.0523
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.1812
- action diff / real delta: 0.162
- noise std / real delta: 0.114
- clamp frac: 0.0131

### zero:cfg1
- target MAE: 0.0408
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.1720
- action diff / real delta: 0.158
- noise std / real delta: 0.000
- clamp frac: 0.0131

## Rollout

### random:cfg1
- mean MAE: 0.0651
- final MAE: 0.0782
- MAE slope/frame: 0.00316
- generated delta / true delta: 0.882
- clamp frac: 0.0139

### zero:cfg1
- mean MAE: 0.0560
- final MAE: 0.0704
- MAE slope/frame: 0.00352
- generated delta / true delta: 0.852
- clamp frac: 0.0140
