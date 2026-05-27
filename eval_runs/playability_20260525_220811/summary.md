# Playability Eval

- NGEN: `diffuse/ngen/runs/ngen_20260210_160745/checkpoints/latest.pt`
- VAE: `diffuse/vae/runs/vae_20260201_111550/checkpoints/latest.pt`
- Device: `mps`
- Windows: 4
- Rollout length: 10
- Euler steps: 8

## One-Step

### random:cfg1
- target MAE: 0.0884
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2098
- action diff / real delta: 0.178
- noise std / real delta: 0.305
- clamp frac: 0.0154

### random:cfg1.5
- target MAE: 0.0898
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2123
- action diff / real delta: 0.192
- noise std / real delta: 0.297
- clamp frac: 0.0149

### zero:cfg1
- target MAE: 0.0601
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.1866
- action diff / real delta: 0.178
- noise std / real delta: 0.000
- clamp frac: 0.0155

### corr:cfg1
- target MAE: 0.0888
- real frame delta MAE: 0.1694
- generated frame delta MAE: 0.2099
- action diff / real delta: 0.177
- noise std / real delta: 0.000
- clamp frac: 0.0154

## Rollout

### random:cfg1
- mean MAE: 0.1122
- final MAE: 0.1289
- MAE slope/frame: 0.00468
- generated delta / true delta: 0.946
- clamp frac: 0.0166

### random:cfg1.5
- mean MAE: 0.1188
- final MAE: 0.1355
- MAE slope/frame: 0.00547
- generated delta / true delta: 0.950
- clamp frac: 0.0169

### zero:cfg1
- mean MAE: 0.0891
- final MAE: 0.1073
- MAE slope/frame: 0.00527
- generated delta / true delta: 0.861
- clamp frac: 0.0169

### corr:cfg1
- mean MAE: 0.1136
- final MAE: 0.1294
- MAE slope/frame: 0.00458
- generated delta / true delta: 0.822
- clamp frac: 0.0161
