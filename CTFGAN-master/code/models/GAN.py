"""CTFGAN: seven fusion scales, conditional discriminator, explicit contour input."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .baseline import Affine, D_Block


class TFBlock(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.affine1 = Affine(cond_dim, channels)
        self.affine2 = Affine(cond_dim, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, condition):
        x = F.relu(self.affine1(x, condition))
        return self.conv(F.relu(self.affine2(x, condition)))


class CFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x, contour):
        contour = F.interpolate(contour, size=x.shape[-2:], mode='nearest')
        return F.relu(x + self.conv(contour))


class UPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, upsample=True,
                 use_tf=True, use_cf=True):
        super().__init__()
        self.upsample = upsample
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.residual = nn.Sequential(nn.ReLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                      nn.ReLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.tf = TFBlock(cond_dim, out_ch) if use_tf else None
        self.cf = CFBlock(out_ch) if use_cf else None

    def forward(self, x, condition, contour):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.shortcut(x) + self.residual(x)
        if self.tf is not None:
            x = self.tf(x, condition)
        if self.cf is not None:
            x = self.cf(x, contour)
        return x


class NetG(nn.Module):
    def __init__(self, ngf=32, nz=100, cond_dim=256, imsize=256, ch_size=3,
                 use_tf=True, use_cf=True):
        super().__init__()
        if imsize < 16 or imsize & (imsize - 1):
            raise ValueError('imsize must be a power of two >= 16')
        self.nz, self.cond_dim, self.imsize = nz, cond_dim, imsize
        self.ngf = ngf
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        resolutions = [2 ** i for i in range(2, int(math.log2(imsize)) + 1)]
        self.resolutions = resolutions
        channels = [ngf * min(256 // r, 8) for r in resolutions]
        self.GBlocks = nn.ModuleList()
        in_ch = ngf * 8
        for i, out_ch in enumerate(channels):
            self.GBlocks.append(UPBlock(in_ch, out_ch, nz + cond_dim, i > 0, use_tf, use_cf))
            in_ch = out_ch
        self.to_rgb = nn.Sequential(nn.ReLU(), nn.Conv2d(in_ch, ch_size, 3, padding=1), nn.Tanh())

    def forward(self, noise, sentence, contour):
        if contour.ndim != 4 or contour.shape[1] != 1 or contour.shape[0] != noise.shape[0]:
            raise ValueError('contour must be B x 1 x H x W, paired with noise/text')
        condition = torch.cat((noise, sentence), dim=1)
        x = self.fc(noise).view(noise.shape[0], self.ngf * 8, 4, 4)
        for block in self.GBlocks:
            x = block(x, condition, contour)
        return self.to_rgb(x)


class NetD(nn.Module):
    def __init__(self, ndf=32, imsize=256, ch_size=3):
        super().__init__()
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, padding=1)
        channels = [ndf * min(2 ** i, 8) for i in range(int(math.log2(imsize)) - 1)]
        self.DBlocks = nn.ModuleList([D_Block(a, b) for a, b in zip(channels[:-1], channels[1:])])

    def forward(self, image):
        x = self.conv_img(image)
        for block in self.DBlocks:
            x = block(x)
        return x


class NetC(nn.Module):
    """Joint image/text/contour discriminator head D(x,e,c)."""
    def __init__(self, ndf=32, cond_dim=256):
        super().__init__()
        self.contour_encoder = nn.Sequential(nn.Conv2d(1, ndf, 3, padding=1), nn.ReLU(),
                                              nn.Conv2d(ndf, ndf, 3, padding=1))
        self.joint_conv = nn.Sequential(nn.Conv2d(ndf * 9 + cond_dim, ndf * 2, 3, padding=1),
                                        nn.LeakyReLU(.2), nn.Conv2d(ndf * 2, 1, 4))

    def forward(self, features, sentence, contour):
        size = features.shape[-2:]
        contour_features = self.contour_encoder(F.interpolate(contour, size=size, mode='nearest'))
        sentence = sentence[:, :, None, None].expand(-1, -1, *size)
        return self.joint_conv(torch.cat((features, sentence, contour_features), dim=1))
