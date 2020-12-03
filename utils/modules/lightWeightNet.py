import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super(Scale, self).__init__()
        self.scale = Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class AWRU(nn.Module):
    def __init__(self, nf, kernel_size, wn, act=nn.ReLU(True)):
        super(AWRU, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

        self.body = nn.Sequential(
            wn(nn.Conv2d(nf, nf, kernel_size, padding=kernel_size//2)),
            act,
            wn(nn.Conv2d(nf, nf, kernel_size, padding=kernel_size//2)),
        )

    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res


class AWMS(nn.Module):
    def __init__(self, nf, out_chl, wn, act=nn.ReLU(True)):
        super(AWMS, self).__init__()
        self.tail_k3 = wn(nn.Conv2d(nf, nf, 3, padding=3//2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(nf, nf, 5, padding=5//2, dilation=1))
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)
        self.fuse = wn(nn.Conv2d(nf, nf, 3, padding=3 // 2))
        self.act = act
        self.w_conv = wn(nn.Conv2d(nf, out_chl, 3, padding=3//2))

    def forward(self, x):
        x0 = self.scale_k3(self.tail_k3(x))
        x1 = self.scale_k5(self.tail_k5(x))
        cur_x = x0 + x1

        fuse_x = self.act(self.fuse(cur_x))
        out = self.w_conv(fuse_x)

        return out


class LFB(nn.Module):
    def __init__(self, nf, wn, act=nn.ReLU(inplace=True)):
        super(LFB, self).__init__()
        self.b0 = AWRU(nf, 3, wn=wn, act=act)
        self.b1 = AWRU(nf, 3, wn=wn, act=act)
        self.b2 = AWRU(nf, 3, wn=wn, act=act)
        self.b3 = AWRU(nf, 3, wn=wn, act=act)
        self.reduction = wn(nn.Conv2d(nf * 4, nf, 3, padding=3//2))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        res = self.reduction(torch.cat([x0, x1, x2, x3], dim=1))

        return self.res_scale(res) + self.x_scale(x)


class WeightNet(nn.Module):
    def __init__(self, config):
        super(WeightNet, self).__init__()

        in_chl = config.IN_CHANNEL
        nf = config.N_CHANNEL
        n_block = config.RES_BLOCK
        out_chl = config.N_WEIGHT
        scale = config.SCALE

        act = nn.ReLU(inplace=True)
        wn = lambda x: nn.utils.weight_norm(x)

        rgb_mean = torch.FloatTensor([0.4488, 0.4371, 0.4040]).view([1, 3, 1, 1]) 
        self.register_buffer('rgb_mean', rgb_mean)

        self.head = nn.Sequential(
            wn(nn.Conv2d(in_chl, nf, 3, padding=3//2)),
            act,
        )

        body = []
        for i in range(n_block):
            body.append(LFB(nf, wn=wn, act=act))
        self.body = nn.Sequential(*body)

        self.up = nn.Sequential(
            wn(nn.Conv2d(nf, nf * scale ** 2, 3, padding=3//2)),
            act,
            nn.PixelShuffle(upscale_factor=scale)
        )

        self.tail = AWMS(nf, out_chl, wn, act=act)

    def forward(self, x):
        x = x - self.rgb_mean
        x = self.head(x)
        x = self.body(x)
        x = self.up(x)
        out = self.tail(x)

        return out


if __name__ == '__main__':
    from easydict import EasyDict as edict

    config = edict()
    config.IN_CHANNEL = 3
    config.N_CHANNEL = 32
    config.RES_BLOCK = 4
    config.N_WEIGHT = 72
    config.SCALE = 2

    net = WeightNet(config).cuda()

    cnt = 0
    for p in net.parameters():
        cnt += p.numel()
    print(cnt)

    x = torch.randn(1, 3, 32, 32).cuda()
    out = net(x)
    print(out.size())
