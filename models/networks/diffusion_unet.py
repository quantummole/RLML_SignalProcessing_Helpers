
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32, device=timesteps.device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        t = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.block1 = ResBlock(in_ch, out_ch, t_emb_dim)
        self.block2 = ResBlock(out_ch, out_ch, t_emb_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        skip = x
        x = self.pool(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResBlock(out_ch+in_ch, out_ch, t_emb_dim)
        self.block2 = ResBlock(out_ch, out_ch, t_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_layers=4, base=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.nl = num_layers
        for i in range(1,self.nl+1):
            layer = Down(base*(2**(i-1)), base*(2**i), time_dim)
            self.downs.append(layer)       
        self.mid1 = ResBlock(base*(2**self.nl), base*(2**self.nl), time_dim)
        self.mid2 = ResBlock(base*(2**self.nl), base*(2**self.nl), time_dim)
        for i in range(self.nl,0,-1):
            layer = Up(base*(2**i), base*(2**(i-1)), time_dim)
            self.ups.append(layer)
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        x = self.in_conv(x)
        intermediates = []
        for layer in self.downs:
            x, skip = layer(x, t_emb)
            intermediates.append(skip)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        for layer, s in zip(self.ups, reversed(intermediates)):
            x = layer(x, s, t_emb)
        return self.out(x)

if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 64, 64)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t)
    print(out.shape)  # Expected output shape: (1, 3, 64, 64)