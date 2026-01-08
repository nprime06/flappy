import torch
import torch.nn as nn
from embedding import TimeEmbedding
from nn.resblock import ResBlock, DownResBlock, UpResBlock

class ResUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, embed_dim, out_channels=None, num_classes=0): # layers > 0 is number of up/down
        super().__init__()
        self.time_embedding = TimeEmbedding(embed_dim=embed_dim)
        self.class_embedding = nn.Embedding(num_classes+1, embed_dim) if num_classes > 0 else None 
        # num_classes=0 means no classes, c and c_emb are None

        # down: 1 -> h, h -> 2h, ... 2**(num_layers - 2)h -> 2**(num_layers - 1)h
        down_blocks_list = [DownResBlock(in_channels, hidden_channels, embed_dim)]
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

    def forward(self, x, t, c=None): # x: (B, C, H, W), t: (B,), c: (B,)
        t_emb = self.time_embedding(t) # (B, embed_dim)
        if self.class_embedding is not None:
            if c is None:
                c = torch.zeros_like(t, dtype=torch.long, device=x.device) # if c is none, will use class 0 (unconditional)
            c_emb = self.class_embedding(c) # (B, embed_dim)
        else:
            c_emb = None
        
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb, c_emb)
            skip_connections.append(skip)
        x = self.bot(x, t_emb, c_emb)
        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop(), t_emb, c_emb)
        x = self.out_conv(x)
        return x