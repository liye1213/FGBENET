import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lib.pvtv2 import pvt_v2_b2


# -------------------------------
# Laplace Conv for Edge Branch
# -------------------------------
class LaplaceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        laplace_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        laplace_kernel = laplace_kernel.unsqueeze(0).unsqueeze(0)
        laplace_kernel = laplace_kernel.repeat((out_channels, in_channels, 1, 1))
        self.conv.weight = nn.Parameter(laplace_kernel)
        self.conv.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# -------------------------------
# CBAM Attention
# -------------------------------
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return spatial_out * x


# -------------------------------
# Residual Block
# -------------------------------
class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, 3, stride, padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, stride, 1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


# -------------------------------
# Multi-Scale Feature Fusion (MGFM)
# -------------------------------
class MGFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.c1 = nn.Conv2d(in_channel, out_channel, 1)
        self.c2 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.c3 = nn.Conv2d(in_channel, out_channel, 7, padding=3)
        self.c4 = nn.Conv2d(in_channel, out_channel, 11, padding=5)
        self.s1 = nn.Conv2d(out_channel * 4, out_channel, 1)
        self.att = CBAM(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        merged = torch.cat([x1, x2, x3, x4], dim=1)
        merged = self.s1(merged)
        return self.att(merged)


# -------------------------------
# FGGA (Frequency-domain Attention)
# -------------------------------
class FGGA(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 8, gate_reduce: int = 16, dilation: int = 3, attn_dropout: float = 0.0):
        super().__init__()
        assert in_dim % 2 == 0, "FGGA: in_dim must be even."
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.down_dim = in_dim // 2
        assert self.down_dim % num_heads == 0
        self.head_dim = self.down_dim // num_heads

        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, self.down_dim, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(self.down_dim),
            nn.GELU(),
        )

        mid = max(1, self.down_dim // gate_reduce)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(self.down_dim, mid, 1, bias=True),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, self.down_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        init_tau = (self.head_dim ** -0.5)
        self.tau = nn.Parameter(torch.ones(self.num_heads, 1, 1) * init_tau)
        self.drop_attn = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

    @staticmethod
    def _complex_l2_norm(z, dim=-1, eps=1e-6):
        denom = torch.sqrt((z.real ** 2 + z.imag ** 2).sum(dim=dim, keepdim=True) + eps)
        return z / denom

    @staticmethod
    def _complex_softmax(z, dim=-1):
        return torch.complex(F.softmax(z.real, dim=dim), F.softmax(z.imag, dim=dim))

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.reduce(x)
        q = torch.fft.fft2(feat.float())
        k = torch.fft.fft2(feat.float())
        v = torch.fft.fft2(feat.float())
        g = torch.fft.fft2(feat.float())

        q = rearrange(q, "b (h d) H W -> b h d (H W)", h=self.num_heads)
        k = rearrange(k, "b (h d) H W -> b h d (H W)", h=self.num_heads)
        v = rearrange(v, "b (h d) H W -> b h d (H W)", h=self.num_heads)

        q = self._complex_l2_norm(q)
        k = self._complex_l2_norm(k)
        attn = (q @ k.conj().transpose(-2, -1)) * self.tau
        attn = self._complex_softmax(attn)
        attn = self.drop_attn(attn)

        out_f = attn @ v
        out_f = rearrange(out_f, "b h d (H W) -> b (h d) H W", H=h, W=w)
        out_f = torch.fft.ifft2(out_f, dim=(-2, -1)).abs()

        gate = self.channel_gate(g.real) * g
        gate = torch.fft.ifft2(gate, dim=(-2, -1)).abs()

        y = torch.cat([out_f, gate], dim=1)
        return y + x


# -------------------------------
# GBE (Boundary Enhancement)
# -------------------------------
class GBE(nn.Module):
    def __init__(self, in_channels, norm_type="GN", dropout=0.0):
        super().__init__()
        Norm = self._norm_factory(norm_type)

        def SepBottleneck(cin):
            mid = max(1, cin // 8)
            return nn.Sequential(
                nn.Conv2d(cin, mid, 1, bias=False),
                nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
                nn.Conv2d(mid, cin, 1, bias=False),
                Norm(cin),
                nn.GELU(),
            )

        self.stage1 = SepBottleneck(in_channels)
        self.stage2 = SepBottleneck(in_channels)
        self.stage3 = SepBottleneck(in_channels)
        self.post = nn.Sequential(
            Norm(in_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

    @staticmethod
    def _norm_factory(norm_type):
        if norm_type.upper() == "GN":
            return lambda c: nn.GroupNorm(num_groups=max(1, c // 16), num_channels=c)
        elif norm_type.upper() == "IN":
            return lambda c: nn.InstanceNorm2d(c, affine=True)
        else:
            return lambda c: nn.BatchNorm2d(c)

    def forward(self, x):
        residual = x
        a = self.stage1(x)
        a = self.stage2(a)
        b = self.stage3(x)
        gated = self.post(a * b)
        return residual + self.alpha * (gated - residual)


# -------------------------------
# Main Field Network
# -------------------------------
class Field(nn.Module):
    def __init__(self, channel=32, num_classes=2, drop_rate=0.4):
        super().__init__()
        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()
        path = r"/home/lyaya/HBGNet-main/pvt_v2_b2.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.edge_lap = LaplaceConv2d(3, 1)
        self.conv1 = Residual(1, 32)
        self.conv2 = nn.Conv2d(64, 16, 1)
        self.att = CBAM(64)

        self.up1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up3 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True)

        self.proj2 = nn.Conv2d(128, 16, 1)
        self.proj3 = nn.Conv2d(320, 16, 1)
        self.proj4 = nn.Conv2d(512, 16, 1)

        self.multi_fusion = MGFM(64, 64)

        self.out_feature = nn.Conv2d(64, 1, 1)
        self.edge_feature = nn.Conv2d(64, num_classes, 1)
        self.dis_feature = nn.Conv2d(64, 1, 1)

        self.fa1, self.fa2, self.fa3, self.fa4 = FGGA(64), FGGA(128), FGGA(320), FGGA(512)
        self.gbe1, self.gbe2, self.gbe3, self.gbe4 = GBE(64), GBE(128), GBE(320), GBE(512)

    def forward(self, x):
        edge = self.conv1(self.edge_lap(x))
        pvt = self.backbone(x)
        x1, x2, x3, x4 = [self.drop(f) for f in pvt]

        x1, x2, x3, x4 = self.fa1(x1), self.fa2(x2), self.fa3(x3), self.fa4(x4)
        x1, x2, x3, x4 = self.gbe1(x1), self.gbe2(x2), self.gbe3(x3), self.gbe4(x4)

        edge = torch.cat([edge, self.up1(x1)], dim=1)
        edge = self.att(edge)
        edge1 = self.conv2(edge)

        bs2, bs3, bs4 = self.proj2(self.up2(x2)), self.proj3(self.up3(x3)), self.proj4(self.up4(x4))
        ms = torch.cat([edge1, bs2, bs3, bs4], dim=1)
        out = self.multi_fusion(ms)

        edge_out = F.log_softmax(self.edge_feature(edge), dim=1)
        mask_out = self.out_feature(out)
        dis_out = self.dis_feature(out)
        return [mask_out, edge_out, dis_out]


if __name__ == "__main__":
    tensor = torch.randn((2, 3, 256, 256))
    net = Field()
    outputs = net(tensor)
    for o in outputs:
        print(o.shape)
