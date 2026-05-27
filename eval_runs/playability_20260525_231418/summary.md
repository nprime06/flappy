# Playability Eval

- NGEN: `diffuse/ngen/runs/ngen_20260202_003536/checkpoints/latest.pt`
- VAE: `diffuse/vae/runs/vae_20260201_111550/checkpoints/latest.pt`
- Device: `mps`
- Windows: 8
- Rollout length: 10
- Euler steps: 20

## One-Step

### random:cfg1
- target MAE: 0.2047
- real frame delta MAE: 0.2053
- generated frame delta MAE: 0.3258
- action diff / real delta: 0.111
- action diff / real delta, no-flap targets: 0.044
- action diff / real delta, flap targets: 0.178
- noise std / real delta: 0.174
- clamp frac: 0.0336

### zero:cfg1
- target MAE: 0.1972
- real frame delta MAE: 0.2053
- generated frame delta MAE: 0.3220
- action diff / real delta: 0.062
- action diff / real delta, no-flap targets: 0.046
- action diff / real delta, flap targets: 0.079
- noise std / real delta: 0.000
- clamp frac: 0.0340

## Rollout

### random:cfg1
- mean MAE: 0.2081
- final MAE: 0.2037
- MAE slope/frame: 0.00085
- generated delta / true delta: 0.777
- clamp frac: 0.0316

### zero:cfg1
- mean MAE: 0.2054
- final MAE: 0.2032
- MAE slope/frame: 0.00122
- generated delta / true delta: 0.733
- clamp frac: 0.0317
