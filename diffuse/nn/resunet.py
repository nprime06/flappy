import torch
import torch.nn as nn
from .embedding import TimeEmbedding
from .resblock import ResBlock, DownResBlock, UpResBlock

class ResUNet(nn.Module):
    '''
    class is action
    k is amount of context
    '''
    def __init__(self, in_channels, hidden_channels, num_layers, embed_dim, k, num_classes, act_embed_dim, num_aug_bins): 
        super().__init__()
        self.time_embedding = TimeEmbedding(embed_dim=embed_dim) # already activated
        self.class_embedding = nn.Embedding(num_classes + 1, act_embed_dim) # cfg
        self.class_proj = nn.Sequential(nn.SiLU(), nn.Linear(act_embed_dim * k, embed_dim))
        self.class_act = nn.SiLU()
        self.aug_embedding = nn.Embedding(num_aug_bins, embed_dim)
        self.aug_act = nn.SiLU()
        # all these embeddings are one-time

        # down: in * (k+1) -> h, h -> 2h, ... 2**(num_layers - 2)h -> 2**(num_layers - 1)h
        down_blocks_list = [DownResBlock(in_channels * (k + 1), hidden_channels, embed_dim)]
        for i in range(num_layers - 1):
            down_blocks_list.append(DownResBlock(hidden_channels * 2**i, hidden_channels * 2**(i + 1), embed_dim))
        self.down_blocks = nn.ModuleList(down_blocks_list)

        self.bot = ResBlock(hidden_channels * 2**(num_layers - 1), hidden_channels * 2**num_layers, embed_dim)
        self.done_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2**num_layers, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # up: 2**(num_layers)h -> 2**(num_layers - 1)h, ... 2h -> h
        up_blocks_list = []
        for i in range(num_layers):
            up_blocks_list.append(UpResBlock(hidden_channels * 2**(num_layers - i), hidden_channels * 2**(num_layers - i - 1), embed_dim))
        self.up_blocks = nn.ModuleList(up_blocks_list)

        self.out_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x, t, z_cond, c, aug_level):
        # x: (B, C, H, W), t: (B,), z_cond: (B, C * k, H, W), c: (B, k), aug_level: (B,)
        x = torch.cat([x, z_cond], dim=1)

        t_emb = self.time_embedding(t)
        action_emb = self.class_embedding(c)
        action_emb = action_emb.flatten(1)
        c_emb = self.class_act(self.class_proj(action_emb))
        aug_emb = self.aug_act(self.aug_embedding(aug_level))
        # get (B, emb_dim)

        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb, c_emb, aug_emb)
            skip_connections.append(skip)
        x = self.bot(x, t_emb, c_emb, aug_emb)

        done_logit = self.done_head(x)

        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop(), t_emb, c_emb, aug_emb)
        x = self.out_conv(x)

        return x, done_logit