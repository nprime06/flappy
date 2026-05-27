# Playability Eval

- NGEN: `diffuse/ngen/runs/ngen_20260210_211125/checkpoints/latest.pt`
- VAE: `diffuse/vae/runs/vae_20260201_111550/checkpoints/latest.pt`
- Device: `mps`
- Windows: 4
- Rollout length: 10
- Euler steps: 8

## One-Step

### random:cfg1
- target MAE: 0.0889
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2103
- action diff / real delta: 0.173
- noise std / real delta: 0.306
- clamp frac: 0.0152

### random:cfg1.5
- target MAE: 0.0939
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2163
- action diff / real delta: 0.194
- noise std / real delta: 0.317
- clamp frac: 0.0152

### zero:cfg1
- target MAE: 0.0608
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.1871
- action diff / real delta: 0.173
- noise std / real delta: 0.000
- clamp frac: 0.0156

### corr:cfg1
- target MAE: 0.0888
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2103
- action diff / real delta: 0.170
- noise std / real delta: 0.000
- clamp frac: 0.0154

## Rollout

### random:cfg1
- mean MAE: 0.1184
- final MAE: 0.1411
- MAE slope/frame: 0.00552
- generated delta / true delta: 0.987
- clamp frac: 0.0170

### random:cfg1.5
- mean MAE: 0.1324
- final MAE: 0.1673
- MAE slope/frame: 0.00773
- generated delta / true delta: 1.002
- clamp frac: 0.0173

### zero:cfg1
- mean MAE: 0.0990
- final MAE: 0.1262
- MAE slope/frame: 0.00717
- generated delta / true delta: 0.900
- clamp frac: 0.0176

### corr:cfg1
- mean MAE: 0.1243
- final MAE: 0.1502
- MAE slope/frame: 0.00680
- generated delta / true delta: 0.852
- clamp frac: 0.0163
