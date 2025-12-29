import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- small helpers --------------------
def dwise_conv(ch_in, stride=1, use_bn=True, act=True):
    layers = [nn.Conv2d(ch_in, ch_in, 3, stride=stride, padding=1, groups=ch_in, bias=False)]
    if use_bn: layers += [nn.BatchNorm2d(ch_in)]
    if act:    layers += [nn.ReLU6(inplace=True)]
    return nn.Sequential(*layers)

def conv1x1(ch_in, ch_out, use_bn=True, act=True):
    layers = [nn.Conv2d(ch_in, ch_out, 1, bias=False)]
    if use_bn: layers += [nn.BatchNorm2d(ch_out)]
    if act:    layers += [nn.ReLU6(inplace=True)]
    return nn.Sequential(*layers)

def conv3x3(ch_in, ch_out, stride, use_bn=True, act=True):
    layers = [nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False)]
    if use_bn: layers += [nn.BatchNorm2d(ch_out)]
    if act:    layers += [nn.ReLU6(inplace=True)]
    return nn.Sequential(*layers)

# -------------------- MobileNetV2 blocks --------------------
class InvertedBlock(nn.Module):
    """ MobileNetV2 (t-expand → depthwise → linear 1x1 project) """
    def __init__(self, ch_in, ch_out, expand_ratio, stride, use_bn=True):
        super().__init__()
        assert stride in [1, 2]
        hidden = ch_in * expand_ratio
        self.use_res = (stride == 1 and ch_in == ch_out)

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden, use_bn=use_bn, act=True))
        layers += [
            dwise_conv(hidden if expand_ratio != 1 else ch_in, stride=stride, use_bn=use_bn, act=True),
            conv1x1(hidden if expand_ratio != 1 else ch_in, ch_out, use_bn=use_bn, act=False)  # linear bottleneck
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return x + y if self.use_res else y

class InvertedUpBlock(nn.Module):
    """ Upsample ×2 → depthwise → linear 1x1 project (V2 스타일 역방향) """
    def __init__(self, ch_in, ch_out, expand_ratio=6, use_bn=True, mode="nearest"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        hidden = ch_in * expand_ratio
        self.expand = conv1x1(ch_in, hidden, use_bn=use_bn, act=True) if expand_ratio != 1 else nn.Identity()
        self.dw = dwise_conv(hidden if expand_ratio != 1 else ch_in, stride=1, use_bn=use_bn, act=True)
        self.project = conv1x1(hidden if expand_ratio != 1 else ch_in, ch_out, use_bn=use_bn, act=False)

    def forward(self, x):
        x = self.up(x)
        x = self.expand(x) if not isinstance(self.expand, nn.Identity) else x
        x = self.dw(x)
        x = self.project(x)
        return F.relu6(x, inplace=True)

# -------------------- Encoder / Decoder --------------------
class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 encoder with total downsample x16:
      128 -> 64 -> 32 -> 16 -> 8
      224 -> 112 -> 56 -> 28 -> 14
    returns:
        z: [B, latent_ch, H/16, W/16]
    """
    def __init__(self, ch_in=3, width_mult=1.0, use_bn=True, latent_ch=320):
        super().__init__()
        self.use_bn = use_bn

        def C(x):
            return int(max(8, round(x * width_mult)))

        # ✅ change the [6,160,3,2] stage stride from 2 -> 1
        self.cfg = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 1],   # <-- was 2, now 1  (removes one downsample)
            [6, 320, 1, 1],
        ]

        # stem (stride 2)
        stem_out = C(32)
        self.stem = conv3x3(ch_in, stem_out, stride=2, use_bn=use_bn, act=True)

        layers = []
        in_ch = stem_out
        for t, c, n, s in self.cfg:
            c = C(c)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(in_ch, c, expand_ratio=t, stride=stride, use_bn=use_bn))
                in_ch = c
        self.features = nn.Sequential(*layers)

        self.latent = conv1x1(in_ch, C(latent_ch), use_bn=use_bn, act=False)
        self.out_ch = C(latent_ch)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        z = self.latent(x)   # now [B, out_ch, H/16, W/16]  -> 128 gives 8x8
        return z
    

class MobileNetV2Decoder(nn.Module):
    """
    Decoder with 4 upsample blocks (x16 total):
      8  -> 16 -> 32 -> 64 -> 128
      14 -> 28 -> 56 -> 112 -> 224
    """
    def __init__(self, in_ch, out_ch=3, width_mult=1.0, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        def C(x):
            return int(max(8, round(x * width_mult)))

        # ✅ 4 upsample blocks (len = 5)
        ch_plan = [in_ch, C(160), C(96), C(64), C(32)]
        ups = []
        for ci, co in zip(ch_plan[:-1], ch_plan[1:]):
            ups.append(InvertedUpBlock(ci, co, expand_ratio=6, use_bn=use_bn, mode="nearest"))
        self.ups = nn.Sequential(*ups)

        self.recon = nn.Sequential(
            dwise_conv(ch_plan[-1], stride=1, use_bn=use_bn, act=True),
            conv1x1(ch_plan[-1], out_ch, use_bn=False, act=False)
        )

    def forward(self, z):
        y = self.ups(z)      # x16 upsample total
        y = self.recon(y)
        return y
    

# -------------------- (Optional) end-to-end wrapper --------------------
class MobileNetV2AutoEncoder(nn.Module):
    def __init__(self, ch_in=3, width_mult=1.0, latent_ch=320, use_bn=True):
        super().__init__()
        self.encoder = MobileNetV2Encoder(ch_in=ch_in, width_mult=width_mult,
                                          use_bn=use_bn, latent_ch=latent_ch)
        self.decoder = MobileNetV2Decoder(in_ch=self.encoder.out_ch, out_ch=ch_in,
                                          width_mult=width_mult, use_bn=use_bn)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

# -------------------- quick test --------------------
if __name__ == "__main__":
    import torch
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    x = torch.randn(1, 3, 128, 128)
    enc = MobileNetV2Encoder(ch_in=3, width_mult=1.0, use_bn=True, latent_ch=128)
    dec = MobileNetV2Decoder(in_ch=enc.out_ch, out_ch=3, width_mult=1.0, use_bn=True)
    enc.eval(); dec.eval()

    with torch.no_grad():
        z = enc(x)

    enc_macs = FlopCountAnalysis(enc, x)
    dec_macs = FlopCountAnalysis(dec, z)

    print(flop_count_table(enc_macs, max_depth=2))
    print(flop_count_table(dec_macs, max_depth=2))


    z = enc(x)
    y = dec(z)
    print("z:", z.shape, "y:", y.shape)  # z: [1, ~320, 8, 8], y: [1, 3, 128, 128]