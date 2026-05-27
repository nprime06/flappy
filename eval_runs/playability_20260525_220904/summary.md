# Playability Eval

- NGEN: `diffuse/ngen/runs/ngen_20260207_005513/checkpoints/latest.pt`
- VAE: `diffuse/vae/runs/vae_20260201_111550/checkpoints/latest.pt`
- Device: `mps`
- Windows: 8
- Rollout length: 10
- Euler steps: 20

## One-Step

### random:cfg1
- target MAE: 0.0499
- real frame delta MAE: 0.1835
- generated frame delta MAE: 0.1954
- action diff / real delta: 0.116
- action diff / real delta, no-flap targets: 0.075
- action diff / real delta, flap targets: 0.157
- noise std / real delta: 0.101
- clamp frac: 0.0133

### zero:cfg1
- target MAE: 0.0390
- real frame delta MAE: 0.1835
- generated frame delta MAE: 0.1868
- action diff / real delta: 0.116
- action diff / real delta, no-flap targets: 0.075
- action diff / real delta, flap targets: 0.156
- noise std / real delta: 0.000
- clamp frac: 0.0134

## Rollout

### random:cfg1
- mean MAE: 0.0604
- final MAE: 0.0854
- MAE slope/frame: 0.00405
- generated delta / true delta: 0.859
- clamp frac: 0.0128

### zero:cfg1
- mean MAE: 0.0475
- final MAE: 0.0768
- MAE slope/frame: 0.00421
- generated delta / true delta: 0.810
- clamp frac: 0.0127
