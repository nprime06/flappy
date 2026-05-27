# Playability Eval

- NGEN: `diffuse/ngen/runs/ngen_20260207_005513/checkpoints/latest.pt`
- VAE: `diffuse/vae/runs/vae_20260201_111550/checkpoints/latest.pt`
- Device: `mps`
- Windows: 2
- Rollout length: 5
- Euler steps: 2

## One-Step

### random:cfg1
- target MAE: 0.3192
- real frame delta MAE: 0.2089
- generated frame delta MAE: 0.4645
- action diff / real delta: 0.145
- noise std / real delta: 0.000
- clamp frac: 0.0260

### zero:cfg1
- target MAE: 0.1361
- real frame delta MAE: 0.2089
- generated frame delta MAE: 0.2975
- action diff / real delta: 0.146
- noise std / real delta: 0.000
- clamp frac: 0.0267

## Rollout

### random:cfg1
- mean MAE: 0.3268
- final MAE: 0.3339
- MAE slope/frame: 0.00393
- generated delta / true delta: 1.250
- clamp frac: 0.0249

### zero:cfg1
- mean MAE: 0.1479
- final MAE: 0.1565
- MAE slope/frame: 0.00523
- generated delta / true delta: 0.849
- clamp frac: 0.0251
