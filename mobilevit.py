import torch
import torch.nn as nn
from einops import rearrange

# -------------------------
# Small building blocks
# -------------------------
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        # x: (B, P, N, C)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        B, P, N, _ = q.shape
        H = self.heads
        def split(t):
            t = t.view(B, P, N, H, -1).transpose(2, 3)  # (B,P,H,N,d)
            return t
        q, k, v = map(split, (q, k, v))
        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = attn @ v  # (B,P,H,N,d)
        out = out.transpose(2, 3).contiguous().view(B, P, N, -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    """MobileNetV2 inverted residual block (stride 1 or 2)."""
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res = (stride == 1 and inp == oup)
        layers = []
        # pw
        layers += [
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        ]
        # dw
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        ]
        # pw-linear
        layers += [
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            out = x + out
        return out

class MobileViTBlock(nn.Module):
    """
    Local conv -> 1x1 -> Transformer (patchified) -> 1x1 -> local conv, then fuse with input via conv
    """
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size, stride=1)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(channel, channel, kernel_size, stride=1)

    def forward(self, x):
        # x: (B,C,H,W); H,W must be divisible by ph,pw
        y = x
        x = self.conv1(x)
        x = self.conv2(x)
        B, D, H, W = x.shape
        assert H % self.ph == 0 and W % self.pw == 0, "Feature size must be divisible by patch size"
        # (B,D,H,W) -> (B, ph*pw, (H*W)/(ph*pw), D)
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=H//self.ph, w=W//self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = x + y  # lightweight residual fusion (no concat)
        x = self.conv4(x)
        return x

# -------------------------
# Encoder / Decoder
# -------------------------
class MobileViTEncoder(nn.Module):
    def __init__(self, image_size=(256,256), cfg=None, expansion=4, kernel_size=3, patch_size=(2,2)):
        super().__init__()
        self.stem = conv_nxn_bn(3, cfg["c0"], kernel_size, stride=2)  # /2

        self.stage1 = nn.Sequential(
            MV2Block(cfg["c0"], cfg["c1"], stride=1, expansion=expansion),
            MV2Block(cfg["c1"], cfg["c2"], stride=2, expansion=expansion),  # /4
        )
        self.stage2 = nn.Sequential(
            MV2Block(cfg["c2"], cfg["c3"], stride=1, expansion=expansion),
            MobileViTBlock(dim=cfg["d1"], depth=cfg["L1"], channel=cfg["c3"],
                           kernel_size=kernel_size, patch_size=patch_size,
                           mlp_dim=cfg["d1"]*2),
        )
        self.stage3 = nn.Sequential(
            MV2Block(cfg["c3"], cfg["c4"], stride=2, expansion=expansion),  # /8
            MobileViTBlock(dim=cfg["d2"], depth=cfg["L2"], channel=cfg["c4"],
                           kernel_size=kernel_size, patch_size=patch_size,
                           mlp_dim=cfg["d2"]*4),
        )
        self.stage4 = nn.Sequential(
            MV2Block(cfg["c4"], cfg["c5"], stride=2, expansion=expansion),  # /16
            MobileViTBlock(dim=cfg["d3"], depth=cfg["L3"], channel=cfg["c5"],
                           kernel_size=kernel_size, patch_size=patch_size,
                           mlp_dim=cfg["d3"]*4),
        )
        # ⚠️ stage5 제거 (더 이상 /32로 내려가지 않음)
        self.cb = cfg["c5"]  # bottleneck channels

    def forward(self, x):
        x = self.stem(x)    # /2
        x = self.stage1(x)  # /4
        x = self.stage2(x)  # /4
        x = self.stage3(x)  # /8
        x = self.stage4(x)  # /16
        return x            # (B, c5, H/16, W/16)

class UpBlock(nn.Module):
    """x2 upsample + conv refinement + (optional) MobileViTBlock."""
    def __init__(self, inp, out, use_mvit=False, d=0, L=0, kernel_size=3, patch_size=(2,2)):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_nxn_bn(inp, out, kernel_size=3, stride=1),
            MV2Block(out, out, stride=1, expansion=2),
        )
        self.use_mvit = use_mvit
        if use_mvit:
            self.mvit = MobileViTBlock(dim=d, depth=L, channel=out,
                                       kernel_size=kernel_size, patch_size=patch_size,
                                       mlp_dim=max(d*2, 32))
        else:
            self.mvit = nn.Identity()

    def forward(self, x):
        x = self.up(x)
        x = self.mvit(x)
        return x

class MobileViTDecoder(nn.Module):
    def __init__(self, image_size=(256,256), cfg=None, kernel_size=3, patch_size=(2,2)):
        super().__init__()
        # /16 -> /8
        self.up1 = UpBlock(cfg["c5"], cfg["c4"], use_mvit=True,  d=cfg["d3"], L=cfg["L3"],
                           kernel_size=kernel_size, patch_size=patch_size)
        # /8 -> /4
        self.up2 = UpBlock(cfg["c4"], cfg["c3"], use_mvit=True,  d=cfg["d2"], L=cfg["L2"],
                           kernel_size=kernel_size, patch_size=patch_size)
        # /4 -> /2
        self.up3 = UpBlock(cfg["c3"], cfg["c2"], use_mvit=True,  d=cfg["d1"], L=cfg["L1"],
                           kernel_size=kernel_size, patch_size=patch_size)
        # /2 -> /
        self.up4 = UpBlock(cfg["c2"], cfg["c1"], use_mvit=False,
                           kernel_size=kernel_size, patch_size=patch_size)

        self.tail = nn.Sequential(
            conv_nxn_bn(cfg["c1"], cfg["c0"], kernel_size=3, stride=1),
            nn.Conv2d(cfg["c0"], 3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.up1(x)  # /8
        x = self.up2(x)  # /4
        x = self.up3(x)  # /2
        x = self.up4(x)  # /
        x = self.tail(x)
        return x

# -------------------------
# Full Autoencoder
# -------------------------
class MobileViT_AE(nn.Module):
    """
    MobileViT-style bottleneck autoencoder (no skip connections).
    """
    def __init__(self, image_size=(256,256), variant="xxs"):
        super().__init__()
        assert variant in ["xxs","xs","s"]
        # Configs chosen to be consistent & simple
        if variant == "xxs":
            cfg = dict(
                # channels
                c0=16, c1=16, c2=24, c3=48, c4=64, c5=80, cb=160,
                # ViT dims & depths
                d1=64, d2=80, d3=96, L1=2, L2=4, L3=3,
            )
            expansion = 2
        elif variant == "xs":
            cfg = dict(
                c0=16, c1=32, c2=48, c3=64, c4=80, c5=96, cb=192,
                d1=96, d2=120, d3=144, L1=2, L2=4, L3=3,
            )
            expansion = 4
        else:  # "s"
            cfg = dict(
                c0=16, c1=32, c2=64, c3=96, c4=128, c5=160, cb=320,
                d1=144, d2=192, d3=240, L1=2, L2=4, L3=3,
            )
            expansion = 4

        self.encoder = MobileViTEncoder(
            image_size=image_size, cfg=cfg, expansion=expansion, kernel_size=3, patch_size=(2,2)
        )
        self.decoder = MobileViTDecoder(
            image_size=image_size, cfg=cfg, kernel_size=3, patch_size=(2,2)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# -------------------------
# Utilities / quick test
# -------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    img = torch.randn(1, 3, 128, 128)
    for v in ["xxs","xs","s"]:
        ae = MobileViT_AE(image_size=(128,128), variant=v)
        out, z = ae(img)
        print(f"variant={v}  out={tuple(out.shape)}  z={tuple(z.shape)}  params={count_parameters(ae):,}")
