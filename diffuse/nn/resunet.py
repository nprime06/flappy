import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .embedding import TimeEmbedding
from .resblock import ResBlock, DownResBlock, UpResBlock

class ResUNet(nn.Module):  # context_size = k (number of context latent frames)
    def __init__(self, in_channels, hidden_channels, num_layers, embed_dim, act_embed_dim, num_classes, context_size, num_aug_bins):
        super().__init__()
        self.time_embedding = TimeEmbedding(embed_dim=embed_dim)  # already activated
        self.aug_embedding = nn.Embedding(num_aug_bins, embed_dim)
        self.aug_act = nn.SiLU()

        # target action spatial conditioning: embed -> project -> broadcast to (B, act_spatial_channels, H, W)
        self.act_spatial_channels = act_embed_dim  # use act_embed_dim as spatial channel count
        self.class_embedding = nn.Embedding(num_classes, act_embed_dim)
        self.class_spatial_proj = nn.Linear(act_embed_dim, self.act_spatial_channels)

        # down: in * (k+1) + act_spatial -> h, h -> 2h, ... 2**(num_layers - 2)h -> 2**(num_layers - 1)h
        first_in_channels = in_channels * (context_size + 1) + self.act_spatial_channels
        down_blocks_list = [DownResBlock(first_in_channels, hidden_channels, embed_dim)]
        for i in range(num_layers - 1):
            down_blocks_list.append(DownResBlock(hidden_channels * 2**i, hidden_channels * 2**(i + 1), embed_dim))
        self.down_blocks = nn.ModuleList(down_blocks_list)

        self.bot = ResBlock(hidden_channels * 2**(num_layers - 1), hidden_channels * 2**num_layers, embed_dim)
        bot_channels = hidden_channels * 2**num_layers
        self.bot_attn_norm = nn.GroupNorm(1, bot_channels)
        self.bot_attn = nn.MultiheadAttention(bot_channels, num_heads=8, batch_first=True)
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
        # x: (B, C, H, W), t: (B,), z_cond: (B, C * k, H, W), c: (B, k+1), aug_level: (B,)
        # c contains k+1 actions: [a_{t-k}, ..., a_{t-1}, a_t] where a_t causes the target
        B, C, H, W = x.shape

        # extract target action (last in sequence) and broadcast spatially
        target_action = c[:, -1]  # (B,)
        action_emb = self.class_embedding(target_action)  # (B, act_embed_dim)
        action_spatial = self.class_spatial_proj(action_emb)  # (B, act_spatial_channels)
        action_spatial = action_spatial.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)  # (B, act_spatial_channels, H, W)

        # concat: noisy target + context frames + spatial action
        x = torch.cat([x, z_cond, action_spatial], dim=1)

        t_emb = self.time_embedding(t)
        aug_emb = self.aug_act(self.aug_embedding(aug_level))

        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = checkpoint(down_block, x, t_emb, aug_emb, use_reentrant=False)
            skip_connections.append(skip)
        x = checkpoint(self.bot, x, t_emb, aug_emb, use_reentrant=False)

        # self-attention at bottleneck
        B_attn, C_attn, H_attn, W_attn = x.shape
        x_flat = self.bot_attn_norm(x).reshape(B_attn, C_attn, -1).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.bot_attn(x_flat, x_flat, x_flat)
        x = x + attn_out.permute(0, 2, 1).reshape(B_attn, C_attn, H_attn, W_attn)

        done_logit = self.done_head(x.detach())

        for up_block in self.up_blocks:
            x = checkpoint(up_block, x, skip_connections.pop(), t_emb, aug_emb, use_reentrant=False)
        x = self.out_conv(x)

        return x, done_logit
