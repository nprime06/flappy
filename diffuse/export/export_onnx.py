"""Export VAE and flow model to ONNX for browser inference."""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.ae import VAE
from nn.resunet import ResUNet


class EncoderWrapper(nn.Module):
    """Wrap VAE encoder to return mean and logvar separately (avoid randn_like in graph)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        # x: (B, 3, 144, 144) -> mean: (B, 4, 18, 18), logvar: (B, 4, 18, 18)
        encoded = self.encoder(x)
        mean, logvar = encoded.chunk(2, dim=1)
        return mean, logvar


class DecoderWrapper(nn.Module):
    """Wrap VAE decoder for clean export."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        # z: (B, 4, 18, 18) -> x: (B, 3, 144, 144)
        return self.decoder(z)


class ResUNetWrapper(nn.Module):
    """Wrap ResUNet - returns v_pred and done_logit."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, c, z_cond, aug_level):
        # x: (B, 4, 18, 18) - current latent
        # t: (B,) - timestep in [0, 1]
        # c: (B, k+1) - action sequence
        # z_cond: (B, k*4, 18, 18) - k past frames concatenated
        # aug_level: (B,) - augmentation bin index
        # Returns: v_pred (B, 4, 18, 18), done_logit (B, 1)
        return self.model(x, t, z_cond=z_cond, c=c, aug_level=aug_level)


def load_vae(checkpoint_path, device):
    """Load VAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    vae = VAE(
        image_channels=cfg["image_channels"],
        hidden_channels=cfg["hidden_channels"],
        latent_channels=cfg["latent_channels"],
        num_layers=cfg["num_layers"],
    ).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    return vae, cfg


def load_flow_model(checkpoint_path, device):
    """Load flow model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = ResUNet(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        embed_dim=cfg["embed_dim"],
        act_embed_dim=cfg["act_embed_dim"],
        num_classes=cfg["num_classes"],
        context_size=cfg["context_size"],
        num_aug_bins=cfg["num_aug_bins"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def export_encoder(vae, output_path, device):
    """Export VAE encoder to ONNX."""
    print(f"Exporting encoder to {output_path}")

    encoder = EncoderWrapper(vae.encoder).to(device)
    encoder.eval()

    # Dummy input: (1, 3, 144, 144)
    dummy_x = torch.randn(1, 3, 144, 144, device=device)

    torch.onnx.export(
        encoder,
        dummy_x,
        output_path,
        opset_version=17,
        input_names=["image"],
        output_names=["mean", "logvar"],
        dynamic_axes={
            "image": {0: "batch"},
            "mean": {0: "batch"},
            "logvar": {0: "batch"},
        },
    )
    print(f"  Encoder exported: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def export_decoder(vae, output_path, device):
    """Export VAE decoder to ONNX."""
    print(f"Exporting decoder to {output_path}")

    decoder = DecoderWrapper(vae.decoder).to(device)
    decoder.eval()

    # Dummy input: (1, 4, 18, 18)
    dummy_z = torch.randn(1, 4, 18, 18, device=device)

    torch.onnx.export(
        decoder,
        dummy_z,
        output_path,
        opset_version=17,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={
            "latent": {0: "batch"},
            "image": {0: "batch"},
        },
    )
    print(f"  Decoder exported: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def export_resunet(flow_model, model_cfg, output_path, device):
    """Export ResUNet flow model to ONNX."""
    print(f"Exporting ResUNet to {output_path}")

    resunet = ResUNetWrapper(flow_model).to(device)
    resunet.eval()

    # Dummy inputs
    batch_size = 1
    latent_channels = model_cfg["in_channels"]  # 4
    context_size = model_cfg["context_size"]  # k
    context_channels = context_size * latent_channels  # k * 4
    num_actions = context_size + 1  # k+1 actions
    latent_h, latent_w = 18, 18

    dummy_x = torch.randn(batch_size, latent_channels, latent_h, latent_w, device=device)
    dummy_t = torch.rand(batch_size, device=device)
    dummy_c = torch.zeros(batch_size, num_actions, dtype=torch.long, device=device)
    dummy_z_cond = torch.randn(batch_size, context_channels, latent_h, latent_w, device=device)
    dummy_aug = torch.zeros(batch_size, dtype=torch.long, device=device)

    torch.onnx.export(
        resunet,
        (dummy_x, dummy_t, dummy_c, dummy_z_cond, dummy_aug),
        output_path,
        opset_version=17,
        input_names=["x", "t", "actions", "z_cond", "aug_level"],
        output_names=["v_pred", "done_logit"],
        dynamic_axes={
            "x": {0: "batch"},
            "t": {0: "batch"},
            "actions": {0: "batch"},
            "z_cond": {0: "batch"},
            "aug_level": {0: "batch"},
            "v_pred": {0: "batch"},
            "done_logit": {0: "batch"},
        },
    )
    print(f"  ResUNet exported: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def convert_to_fp16(input_path, output_path):
    """Convert ONNX model to float16."""
    try:
        import onnx
        from onnxconverter_common import float16

        print(f"Converting {input_path} to fp16...")
        model = onnx.load(input_path)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, output_path)
        print(f"  FP16 model: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        return True
    except ImportError:
        print("  Warning: onnxconverter-common not installed. Skipping fp16 conversion.")
        print("  Install with: pip install onnxconverter-common")
        return False


def validate_onnx(onnx_path):
    """Validate ONNX model."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  Validation passed: {onnx_path}")
        return True
    except ImportError:
        print("  Warning: onnx not installed. Skipping validation.")
        return False
    except Exception as e:
        print(f"  Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX for browser inference")
    parser.add_argument("--vae-checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint")
    parser.add_argument("--flow-checkpoint", type=str, required=True,
                        help="Path to flow model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./onnx_models",
                        help="Output directory for ONNX files")
    parser.add_argument("--fp16", action="store_true",
                        help="Convert models to float16")
    parser.add_argument("--validate", action="store_true",
                        help="Validate exported ONNX models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("\nLoading models...")
    vae, vae_cfg = load_vae(args.vae_checkpoint, device)
    flow_model, flow_cfg = load_flow_model(args.flow_checkpoint, device)

    print(f"  VAE config: {vae_cfg}")
    print(f"  Flow config: {flow_cfg}")

    # Export paths
    encoder_path = os.path.join(args.output_dir, "encoder.onnx")
    decoder_path = os.path.join(args.output_dir, "decoder.onnx")
    resunet_path = os.path.join(args.output_dir, "resunet.onnx")

    # Export models
    print("\nExporting models...")
    with torch.no_grad():
        export_encoder(vae, encoder_path, device)
        export_decoder(vae, decoder_path, device)
        export_resunet(flow_model, flow_cfg, resunet_path, device)

    # Validate exports
    if args.validate:
        print("\nValidating exports...")
        validate_onnx(encoder_path)
        validate_onnx(decoder_path)
        validate_onnx(resunet_path)

    # Convert to fp16
    if args.fp16:
        print("\nConverting to fp16...")
        encoder_fp16_path = os.path.join(args.output_dir, "encoder_fp16.onnx")
        decoder_fp16_path = os.path.join(args.output_dir, "decoder_fp16.onnx")
        resunet_fp16_path = os.path.join(args.output_dir, "resunet_fp16.onnx")

        convert_to_fp16(encoder_path, encoder_fp16_path)
        convert_to_fp16(decoder_path, decoder_fp16_path)
        convert_to_fp16(resunet_path, resunet_fp16_path)

        if args.validate:
            print("\nValidating fp16 exports...")
            validate_onnx(encoder_fp16_path)
            validate_onnx(decoder_fp16_path)
            validate_onnx(resunet_fp16_path)

    # Save config for browser
    import json
    config = {
        "latent_mean": 10.1880,
        "latent_std": 13.3726,
        "context_size": flow_cfg["context_size"],
        "latent_channels": flow_cfg["in_channels"],
        "num_aug_bins": flow_cfg["num_aug_bins"],
        "num_classes": flow_cfg["num_classes"],
        "image_size": 144,
        "latent_size": 18,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # Summary
    print("\n" + "=" * 50)
    print("Export Summary:")
    print("=" * 50)
    total_size = 0
    for filename in os.listdir(args.output_dir):
        filepath = os.path.join(args.output_dir, filename)
        size = os.path.getsize(filepath) / 1024 / 1024
        print(f"  {filename}: {size:.2f} MB")
        total_size += size
    print(f"  Total: {total_size:.2f} MB")
    print("\nDone! Copy the ONNX files to web/public/models/")


if __name__ == "__main__":
    main()
