import torch
import torch.nn as nn


def get_groups(channels, max_groups=16):
    """Find largest divisor of channels that is <= max_groups."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module): # expects time conditioning, need to change adaGN for non time conditioning
    def __init__(self, fan_in, fan_out, embed_dim, groups=16):
        super().__init__()
        self.gn1 = nn.GroupNorm(get_groups(fan_in, groups), fan_in, affine=True)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(fan_in, fan_out, kernel_size=3, padding=1)
        if embed_dim is None:
            self.gn2 = nn.GroupNorm(get_groups(fan_out, groups), fan_out, affine=True)
        else:
            self.gn2 = nn.GroupNorm(get_groups(fan_out, groups), fan_out, affine=False) # adaGN time embed on second GN
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(fan_out, fan_out, kernel_size=3, padding=1)
        self.act3 = nn.SiLU()

        if fan_in != fan_out:
            self.skip_conv = nn.Conv2d(fan_in, fan_out, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

        if embed_dim is not None:
            self.time_proj = nn.Linear(embed_dim, fan_out * 2)
            self.class_proj = nn.Linear(embed_dim, fan_out * 2)
            self.aug_proj = nn.Linear(embed_dim, fan_out * 2)
            # Zero-init all projections so conditioning is no-op at start
            for proj in [self.time_proj, self.class_proj, self.aug_proj]:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

    def forward(self, x, t_emb, c_emb, aug_emb=None): # t_emb, c_emb, aug_emb are (B, embed_dim)
        res = self.skip_conv(x)
        x = self.gn2(self.conv1(self.act1(self.gn1(x))))
        if t_emb is not None:
            scale, shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            if c_emb is not None:
                class_scale, class_shift = self.class_proj(c_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
                scale, shift = scale + class_scale, shift + class_shift
            if aug_emb is not None:
                aug_scale, aug_shift = self.aug_proj(aug_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
                scale, shift = scale + aug_scale, shift + aug_shift
            x = x * (scale + 1) + shift
        x = self.conv2(self.act2(x))
        x = x + res
        x = self.act3(x)
        return x

class DownResBlock(nn.Module):
    def __init__(self, fan_in, fan_out, embed_dim):
        super().__init__()
        self.res = ResBlock(fan_in, fan_out, embed_dim)
        self.down = nn.Conv2d(fan_out, fan_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb, c_emb, aug_emb=None, get_skip=True):
        x = self.res(x, t_emb, c_emb, aug_emb)
        if get_skip: skip = x
        x = self.down(x)
        if get_skip: return x, skip
        return x

class UpResBlock(nn.Module):
    def __init__(self, fan_in, fan_out, embed_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(fan_in, fan_out, kernel_size=4, stride=2, padding=1)
        if embed_dim is None:
            self.res = ResBlock(fan_out, fan_out, embed_dim=None)
        else:
            self.res = ResBlock(fan_out * 2, fan_out, embed_dim)

    def forward(self, x, skip, t_emb, c_emb, aug_emb=None):
        x = self.up(x)
        if skip is not None:
            # Handle spatial dimension mismatch from odd input sizes
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb, c_emb, aug_emb)
        return x