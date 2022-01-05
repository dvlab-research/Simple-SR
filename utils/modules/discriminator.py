import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Conv_norm_relu(nn.Module):
    def __init__(self, in_chl, out_chl, ksize, stride, padding, dilation=1, has_bias=False,
                 has_sn=False, norm='bn', has_relu=True):
        super(Conv_norm_relu, self).__init__()

        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation,
                              bias=has_bias)

        if has_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_chl)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_chl)
        else:
            self.norm = lambda x: x

        self.has_relu = has_relu
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.has_relu:
            out = self.relu(out)

        return out


class ConvNet(BaseNetwork):
    def __init__(self, in_chl, nf, depth, has_sn=False, norm='bn', use_tanh=False, init_weights=True):
        super(ConvNet, self).__init__()

        layers = []
        layers.append(Conv_norm_relu(in_chl=in_chl, out_chl=nf, ksize=4, stride=2, padding=1))
        for i in range(depth - 1):
            nf_prev = nf
            nf = min(nf * 2, 256)
            has_relu = True if i != depth - 2 else False
            layers.append(Conv_norm_relu(in_chl=nf_prev, out_chl=nf, ksize=4, stride=2, padding=1, has_sn=has_sn,
                                         norm=norm, has_relu=has_relu))

        self.discriminator = nn.Sequential(*layers)

        self.use_tanh = use_tanh

        if init_weights:
            self.init_weights()

    def forward(self, x):
        out = self.discriminator(x)

        if self.use_tanh:
            out = (torch.tanh(out) + 1) / 2

        return out


class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_chl, nf):
        super(Discriminator_VGG_128, self).__init__()
        # in: [in_chl, 128, 128]
        self.conv0_0 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [nf, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [nf*2, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [nf*4, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 4, 4]
        self.linear1 = nn.Linear(nf * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_chl, nf):
        super(Discriminator_VGG_192, self).__init__()
        # in: [in_chl, 192, 192]
        self.conv0_0 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [nf, 96, 96]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [nf*2, 48, 48]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [nf*4, 24, 24]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 12, 12]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 6, 6]
        self.conv5_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn5_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 3, 3]
        self.linear1 = nn.Linear(nf * 8 * 3 * 3, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
        fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class Discriminator_VGG_patch_192(nn.Module):
    def __init__(self, in_chl, nf):
        super(Discriminator_VGG_patch_192, self).__init__()
        # in: [in_chl, 192, 192]
        self.conv0_0 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [nf, 96, 96]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [nf*2, 48, 48]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [nf*4, 24, 24]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 12, 12]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 6, 6]
        self.conv5_0 = nn.Conv2d(nf * 8, 1, 3, 1, 1, bias=False)
        # self.bn5_0 = nn.BatchNorm2d(nf * 8, affine=True)
        # self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        # self.bn5_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # # [nf*8, 3, 3]
        # self.linear1 = nn.Linear(nf * 8 * 3 * 3, 100)
        # self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = self.conv5_0(fea)
        batch = fea.size(0)
        out = fea.view(batch, -1)
        # fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
        # fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))

        # fea = fea.view(fea.size(0), -1)
        # fea = self.lrelu(self.linear1(fea))
        # out = self.linear2(fea)
        return out


class Discriminator_VGG_256(nn.Module):
    def __init__(self, in_chl, nf):
        super(Discriminator_VGG_256, self).__init__()
        # in: [in_chl, 256, 256]
        self.conv0_0 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [nf, 128, 128]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [nf*2, 64, 64]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [nf*4, 32, 32]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 16, 16]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 8, 8]
        self.conv5_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn5_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 4, 4]
        self.linear1 = nn.Linear(nf * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
        fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


if __name__ == '__main__':
    net = Discriminator_VGG_patch_192(3, 64)
    data = torch.randn(8, 3, 192, 192)
    output = net(data)
    print(output.shape)

