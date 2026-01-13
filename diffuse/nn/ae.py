import torch
import torch.nn as nn
from nn.resblock import ResBlock, DownResBlock, UpResBlock

class Encoder(nn.Module):
    def __init__(self, image_channels, hidden_channels, latent_channels, num_layers):
        super().__init__()
        self.down_blocks = nn.ModuleList([DownResBlock(image_channels, hidden_channels, embed_dim=None)])
        for i in range(num_layers - 1):
            self.down_blocks.append(DownResBlock(hidden_channels * 2**i, hidden_channels * 2**(i + 1), embed_dim=None))
        self.out_block = ResBlock(hidden_channels * 2**(num_layers - 1), 2 * latent_channels, embed_dim=None)

    def forward(self, x): # x: (B, image_channels, H, W) -> (B, 2*latent_channels, H', W')
        for down_block in self.down_blocks:
            x = down_block(x, t_emb=None, c_emb=None, get_skip=False)
        x = self.out_block(x, t_emb=None, c_emb=None)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_channels, hidden_channels, image_channels, num_layers):
        super().__init__()
        self.in_block = ResBlock(latent_channels, hidden_channels * 2**(num_layers - 1), embed_dim=None)
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels * 2 ** max(num_layers - 1 - i, 0)
            out_ch = hidden_channels * 2 ** max(num_layers - 2 - i, 0)
            self.up_blocks.append(UpResBlock(in_ch, out_ch, embed_dim=None))
        self.out_block = nn.Conv2d(hidden_channels, image_channels, kernel_size=1)

    def forward(self, z): # z: (B, latent_channels, H', W') -> (B, image_channels, H, W)
        x = self.in_block(z, t_emb=None, c_emb=None)
        for up_block in self.up_blocks:
            x = up_block(x, skip=None, t_emb=None, c_emb=None)
        x = self.out_block(x)
        return x


class VAE(nn.Module):
    def __init__(self, image_channels, hidden_channels, latent_channels, num_layers):
        super().__init__()
        self.encoder = Encoder(image_channels, hidden_channels, latent_channels, num_layers)
        self.decoder = Decoder(latent_channels, hidden_channels, image_channels, num_layers)

    def reparameterize(self, encoded, sample=True): # encoded: (B, 2*latent_channels, H', W')
        mean, logvar = encoded.chunk(2, dim=1)
        if sample:
            logvar = torch.clamp(logvar, min=-30, max=20)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean