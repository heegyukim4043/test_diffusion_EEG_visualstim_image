# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        convs = []
        c = in_channels
        for i in range(num_layers):
            out_c = hidden_dim
            convs.append(nn.Conv1d(c, out_c, kernel_size=5, padding=2))
            convs.append(nn.GroupNorm(8, out_c))
            convs.append(nn.SiLU())
            convs.append(nn.MaxPool1d(kernel_size=2))
            c = out_c
        self.conv = nn.Sequential(*convs)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, C, T)
        h = self.conv(x)      # (B, hidden_dim, T')
        h = h.mean(dim=-1)    # (B, hidden_dim)
        return self.proj(h)   # (B, hidden_dim)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half_dim = dim // 2
    emb_scale = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb   # (B, dim)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, cond):
        # cond: (B, cond_dim)
        cond_out = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h + cond_out)

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h + cond_out)

        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, downsample=True):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, cond_dim)
        self.res2 = ResBlock(out_channels, out_channels, cond_dim)
        self.downsample = downsample
        if downsample:
            self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, cond):
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        skip = x
        if self.downsample:
            x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, skip_channels, upsample=True):
        super().__init__()
        self.upsample = upsample
        if upsample:
            # 인코더에서 올라온 feature upsample
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
            res_in = out_channels + skip_channels
        else:
            self.up = nn.Identity()
            res_in = in_channels + skip_channels

        self.res1 = ResBlock(res_in, out_channels, cond_dim)
        self.res2 = ResBlock(out_channels, out_channels, cond_dim)

    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # (B, res_in, H, W)
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        return x




class EEGConditionalUNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, cond_dim=256):
        super().__init__()
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels, cond_dim, downsample=True)  # out: 64
        self.down2 = DownBlock(base_channels, base_channels * 2, cond_dim, downsample=True)  # out: 128
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, cond_dim, downsample=True)  # out: 256

        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, cond_dim)  # 256 -> 256
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, cond_dim)

        # skip 채널 수를 명시적으로 전달
        self.up3 = UpBlock(
            in_channels=base_channels * 4,  # 256 (mid 출력)
            out_channels=base_channels * 2,  # 128
            cond_dim=cond_dim,
            skip_channels=base_channels * 4,  # skip3: 256
            upsample=True,
        )
        self.up2 = UpBlock(
            in_channels=base_channels * 2,  # 128
            out_channels=base_channels,  # 64
            cond_dim=cond_dim,
            skip_channels=base_channels * 2,  # skip2: 128
            upsample=True,
        )
        self.up1 = UpBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            cond_dim=cond_dim,
            skip_channels=base_channels,
            upsample=True,
        )

        self.final_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, cond):
        x = self.init_conv(x)
        x, s1 = self.down1(x, cond)
        x, s2 = self.down2(x, cond)
        x, s3 = self.down3(x, cond)

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        x = self.up3(x, s3, cond)
        x = self.up2(x, s2, cond)
        x = self.up1(x, s1, cond)

        return self.final_conv(x)


class EEGDiffusionModel(nn.Module):
    def __init__(
        self,
        img_size=64,
        img_channels=3,
        eeg_channels=32,
        eeg_hidden_dim=256,
        time_dim=256,
        base_channels=64,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_timesteps = num_timesteps

        self.eeg_encoder = EEGEncoder(eeg_channels, eeg_hidden_dim)
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_proj = nn.Linear(time_dim + eeg_hidden_dim, time_dim + eeg_hidden_dim)

        self.unet = EEGConditionalUNet(
            img_channels=img_channels,
            base_channels=base_channels,
            cond_dim=time_dim + eeg_hidden_dim,
        )

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def get_cond(self, t, eeg):
        # t: (B,), eeg: (B, C, T)
        t_emb = sinusoidal_embedding(t, self.time_dim)   # (B, time_dim)
        t_emb = self.time_mlp(t_emb)
        eeg_feat = self.eeg_encoder(eeg)                 # (B, eeg_hidden_dim)
        cond = torch.cat([t_emb, eeg_feat], dim=-1)      # (B, time_dim+eeg_hidden_dim)
        cond = self.cond_proj(cond)
        return cond

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x_start, eeg, t):
        cond = self.get_cond(t, eeg)
        x_noisy = self.q_sample(x_start, t)
        eps_pred = self.unet(x_noisy, cond)
        return eps_pred

    def p_losses(self, x_start, eeg, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        cond = self.get_cond(t, eeg)
        eps_pred = self.unet(x_noisy, cond)
        loss = F.mse_loss(eps_pred, noise)
        return loss
