import torch
import torch.nn as nn
from nn.embedding import TimeEmbedding
from nn.resblock import ResBlock, DownResBlock, UpResBlock

class ResUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, embed_dim, out_channels=None, num_classes=0, context_channels=0, num_aug_bins=0):
        super().__init__()
        self.context_channels = context_channels
        self.time_embedding = TimeEmbedding(embed_dim=embed_dim)
        self.class_embedding = nn.Embedding(num_classes + 1, embed_dim) if num_classes > 0 else None
        self.aug_embedding = nn.Embedding(num_aug_bins, embed_dim) if num_aug_bins > 0 else None

        # down: (in + context) -> h, h -> 2h, ... 2**(num_layers - 2)h -> 2**(num_layers - 1)h
        down_blocks_list = [DownResBlock(in_channels + context_channels, hidden_channels, embed_dim)]
        for i in range(num_layers - 1):
            down_blocks_list.append(DownResBlock(hidden_channels * 2**i, hidden_channels * 2**(i + 1), embed_dim))
        self.down_blocks = nn.ModuleList(down_blocks_list)

        self.bot = ResBlock(hidden_channels * 2**(num_layers - 1), hidden_channels * 2**num_layers, embed_dim)

        # up: 2**(num_layers)h -> 2**(num_layers - 1)h, ... 2h -> h
        up_blocks_list = []
        for i in range(num_layers):
            up_blocks_list.append(UpResBlock(hidden_channels * 2**(num_layers - i), hidden_channels * 2**(num_layers - i - 1), embed_dim))
        self.up_blocks = nn.ModuleList(up_blocks_list)

        if out_channels is None:
            out_channels = in_channels
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x, t, c=None, z_cond=None, aug_level=None):
        # x: (B, C, H, W), t: (B,), c: (B,), z_cond: (B, context_channels, H, W), aug_level: (B,)
        if z_cond is not None:
            x = torch.cat([x, z_cond], dim=1)

        t_emb = self.time_embedding(t)
        c_emb = self.class_embedding(c) if self.class_embedding is not None and c is not None else None
        aug_emb = self.aug_embedding(aug_level) if self.aug_embedding is not None and aug_level is not None else None

        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb, c_emb, aug_emb)
            skip_connections.append(skip)
        x = self.bot(x, t_emb, c_emb, aug_emb)
        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop(), t_emb, c_emb, aug_emb)
        x = self.out_conv(x)
        return x