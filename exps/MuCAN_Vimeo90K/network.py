import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../..')
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation

from utils.modules.module_util import ResidualBlock_noBN_noAct, make_layer
from utils.loss import CharbonnierLoss


class Conv_relu(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size, stride, padding, has_relu=True, efficient=False):
        super(Conv_relu, self).__init__()
        self.has_relu = has_relu
        self.efficient = efficient

        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        def _func_factory(conv, relu, has_relu):
            def func(x):
                x = conv(x)
                if has_relu:
                    x = relu(x)
                return x
            return func

        func = _func_factory(self.conv, self.relu, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x


class Attention(nn.Module):
    def __init__(self, nf=64):
        super(Attention, self).__init__()
        self.sAtt_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        att = self.lrelu(self.sAtt_1(x))
        att_max = self.max_pool(att)
        att_avg = self.avg_pool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))

        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.max_pool(att_L)
        att_avg = self.avg_pool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        out = x * att * 2 + att_add

        return out


class CNCAM(nn.Module):
    def __init__(self, nf=64, n_level=3):
        super(CNCAM, self).__init__()
        self.nf = nf
        self.nl = n_level
        self.down_conv = Conv_relu(nf, nf, 3, 2, 1, has_relu=True)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.cat_conv = Conv_relu(nf * self.nl + nf, nf * 4, 1, 1, 0, has_relu=True)
        self.ps = nn.PixelShuffle(2)
        self.up_conv1 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.up_conv2 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)

        self.spa_l = nn.ModuleList()
        for i in range(self.nl + 1):
            self.spa_l.append(Attention(nf))

    def forward(self, x):
        down_fea = self.down_conv(x)
        B, C, H, W = down_fea.size()
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W).cuda()

        p_list = list()
        for j in range(self.nl):
            if p_list:
                p_list.append(self.avg_pool(p_list[-1]))
            else:
                p_list.append(self.avg_pool(down_fea))

        # size of query: B, H * W, C
        query = down_fea.view(B, C, H * W).permute(0, 2, 1)
        query = F.normalize(query, p=2, dim=2)

        keys = list()
        for j in range(self.nl):
            keys.append(F.normalize(p_list[j].view(B, C, -1), p=2, dim=1))

        att_fea = self.spa_l[0](down_fea)
        all_f = [att_fea]
        for j in range(self.nl):
            sim = torch.matmul(query, keys[j])
            ind = sim.argmax(dim=2).view(-1)
            sim_f = keys[j][ind_B, :, ind].view(B, H, W, C).permute(0, 3, 1, 2)
            att_sim_f = self.spa_l[j + 1](sim_f)
            all_f.append(att_sim_f)

        all_f = torch.cat(all_f, dim=1)
        cat_fea = self.cat_conv(all_f)
        up_fea = self.ps(cat_fea)
        up_fea = self.up_conv1(up_fea)

        fea = torch.cat([x, up_fea], dim=1)
        out = self.up_conv2(fea)

        return out


class Aggregate(nn.Module):
    def __init__(self, nf=64, nbr=2, n_group=8, kernels=[3, 3, 3, 3], patches=[7, 11, 15], cor_ksize=3):
        super(Aggregate, self).__init__()
        self.nbr = nbr
        self.cas_k = kernels[0]
        self.k1 = kernels[1]
        self.k2 = kernels[2]
        self.k3 = kernels[3]
        self.g = n_group

        self.L3_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L3_conv2 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.L3_conv3 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.L3_conv4 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.L3_mask = Conv_relu(nf, self.g * self.k3 ** 2, self.k3, 1, (self.k3-1)//2, has_relu=False)
        self.L3_nn_conv = Conv_relu(nf * self.nbr, nf, 3, 1, 1, has_relu=True)

        self.L2_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L2_conv2 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L2_conv3 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.L2_conv4 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.L2_conv5 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.L2_mask = Conv_relu(nf, self.g * self.k2 ** 2, self.k2, 1, (self.k2-1)//2, has_relu=False)
        self.L2_nn_conv = Conv_relu(nf * self.nbr, nf, 3, 1, 1, has_relu=True)
        self.L2_fea_conv = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)

        self.L1_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L1_conv2 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L1_conv3 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.L1_conv4 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.L1_conv5 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.L1_mask = Conv_relu(nf, self.g * self.k1 ** 2, self.k1, 1, (self.k1-1)//2, has_relu=False)
        self.L1_nn_conv = Conv_relu(nf * self.nbr, nf, 3, 1, 1, has_relu=True)
        self.L1_fea_conv = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)

        self.cas_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.cas_conv2 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.cas_conv3 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.cas_conv4 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.cas_mask = Conv_relu(nf, self.g * self.cas_k ** 2, self.cas_k, 1, (self.cas_k-1)//2, has_relu=False)
        self.cas_nn_conv = Conv_relu(nf * self.nbr, nf, 3, 1, 1, has_relu=True)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.patch_size = patches
        self.cor_k = cor_ksize
        self.padding = (self.cor_k - 1) // 2
        self.pad_size = [self.padding + (patch - 1) // 2 for patch in self.patch_size]
        self.add_num = [2 * pad - self.cor_k + 1 for pad in self.pad_size]
        self.L3_corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch_size[0],
                stride=1, padding=self.padding, dilation=1, dilation_patch=1)
        self.L2_corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch_size[1],
                stride=1, padding=self.padding, dilation=1, dilation_patch=1)
        self.L1_corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch_size[2],
                stride=1, padding=self.padding, dilation=1, dilation_patch=1)

    def forward(self, nbr_fea_l, ref_fea_l):
        # L3
        B, C, H, W = nbr_fea_l[2].size()
        L3_w = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_w = self.L3_conv4(self.L3_conv3(self.L3_conv2(self.L3_conv1(L3_w))))
        L3_mask = self.L3_mask(L3_w).view(B, self.g, 1, self.k3 ** 2, H, W)
        # corr: B, (2 * dis + 1) ** 2, H, W
        L3_norm_ref_fea = F.normalize(ref_fea_l[2], dim=1)
        L3_norm_nbr_fea = F.normalize(nbr_fea_l[2], dim=1)
        L3_corr = self.L3_corr(L3_norm_ref_fea, L3_norm_nbr_fea).view(B, -1, H, W)
        # corr_ind: B, H, W
        _, L3_corr_ind = torch.topk(L3_corr, self.nbr, dim=1)
        L3_corr_ind = L3_corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        L3_ind_row_add = L3_corr_ind // self.patch_size[0] * (W + self.add_num[0])
        L3_ind_col_add = L3_corr_ind % self.patch_size[0]
        L3_corr_ind = L3_ind_row_add + L3_ind_col_add
        # generate top-left indexes
        y = torch.arange(H).repeat_interleave(W).cuda()
        x = torch.arange(W).repeat(H).cuda()
        L3_lt_ind = y * (W + self.add_num[0]) + x
        L3_lt_ind = L3_lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        L3_corr_ind = (L3_corr_ind + L3_lt_ind).view(-1)
        # L3_nbr: B, 64 * k * k, (H + 2 * pad - k + 1) * (W + 2 * pad -k + 1)
        L3_nbr = F.unfold(nbr_fea_l[2], self.cor_k, dilation=1, padding=self.pad_size[0], stride=1)
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W * self.nbr).cuda()
        # L3: B * H * W * nbr, 64 * k * k
        L3 = L3_nbr[ind_B, :, L3_corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        L3 = self.L3_nn_conv(L3)
        L3 = L3.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        L3 = L3.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        L3 = self.relu((L3 * L3_mask).sum(dim=3).view(B, C, H, W))

        # L2
        B, C, H, W = nbr_fea_l[1].size()
        L2_w = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_w = self.L2_conv1(L2_w)
        L3_w = F.interpolate(L3_w, scale_factor=2, mode='bilinear', align_corners=False)
        L2_w = self.L2_conv5(self.L2_conv4(self.L2_conv3(self.L2_conv2(torch.cat([L2_w, L3_w], dim=1)))))
        L2_mask = self.L2_mask(L2_w).view(B, self.g, 1, self.k2 ** 2, H, W)
        # generate most similar feas
        L2_norm_ref_fea = F.normalize(ref_fea_l[1], dim=1)
        L2_norm_nbr_fea = F.normalize(nbr_fea_l[1], dim=1)
        L2_corr = self.L2_corr(L2_norm_ref_fea, L2_norm_nbr_fea).view(B, -1, H, W)
        _, L2_corr_ind = torch.topk(L2_corr, self.nbr, dim=1)
        L2_corr_ind = L2_corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        L2_ind_row_add = L2_corr_ind // self.patch_size[1] * (W + self.add_num[1])
        L2_ind_col_add = L2_corr_ind % self.patch_size[1]
        L2_corr_ind = L2_ind_row_add + L2_ind_col_add
        # generate top-left indexes
        y = torch.arange(H).repeat_interleave(W).cuda()
        x = torch.arange(W).repeat(H).cuda()
        L2_lt_ind = y * (W + self.add_num[1]) + x
        L2_lt_ind = L2_lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        L2_corr_ind = (L2_corr_ind + L2_lt_ind).view(-1)
        L2_nbr = F.unfold(nbr_fea_l[1], self.cor_k, dilation=1, padding=self.pad_size[1], stride=1)
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W * self.nbr).cuda()
        L2 = L2_nbr[ind_B, :, L2_corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        L2 = self.L2_nn_conv(L2)
        L2 = L2.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        L2 = L2.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        L2 = self.relu((L2 * L2_mask).sum(dim=3).view(B, C, H, W))
        # fuse F2 with F3
        L3 = F.interpolate(L3, scale_factor=2, mode='bilinear', align_corners=False)
        L2 = self.L2_fea_conv(torch.cat([L2, L3], dim=1))

        # L1
        B, C, H, W = nbr_fea_l[0].size()
        L1_w = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_w = self.L1_conv1(L1_w)
        L2_w = F.interpolate(L2_w, scale_factor=2, mode='bilinear', align_corners=False)
        L1_w = self.L1_conv5(self.L1_conv4(self.L1_conv3(self.L1_conv2(torch.cat([L1_w, L2_w], dim=1)))))
        L1_mask = self.L1_mask(L1_w).view(B, self.g, 1, self.k1 ** 2, H, W)
        # generate mot similar feas
        L1_norm_ref_fea = F.normalize(ref_fea_l[0], dim=1)
        L1_norm_nbr_fea = F.normalize(nbr_fea_l[0], dim=1)
        L1_corr = self.L1_corr(L1_norm_ref_fea, L1_norm_nbr_fea).view(B, -1, H, W)
        _, L1_corr_ind = torch.topk(L1_corr, self.nbr, dim=1)
        L1_corr_ind = L1_corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        L1_ind_row_add = L1_corr_ind // self.patch_size[2] * (W + self.add_num[2])
        L1_ind_col_add = L1_corr_ind % self.patch_size[2]
        L1_corr_ind = L1_ind_row_add + L1_ind_col_add
        # generate top-left indexes
        y = torch.arange(H).repeat_interleave(W).cuda()
        x = torch.arange(W).repeat(H).cuda()
        L1_lt_ind = y * (W + self.add_num[2]) + x
        L1_lt_ind = L1_lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        L1_corr_ind = (L1_corr_ind + L1_lt_ind).view(-1)
        L1_nbr = F.unfold(nbr_fea_l[0], self.cor_k, dilation=1, padding=self.pad_size[2], stride=1)
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W * self.nbr).cuda()
        L1 = L1_nbr[ind_B, :, L1_corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        # L1 = L1.permute(0, 2, 1, 3, 4).contiguous().view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        L1 = self.L1_nn_conv(L1)
        L1 = L1.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        L1 = L1.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        L1 = self.relu((L1 * L1_mask).sum(dim=3).view(B, C, H, W))
        # fuse L1 with L2
        L2 = F.interpolate(L2, scale_factor=2, mode='bilinear', align_corners=False)
        L1 = self.L1_fea_conv(torch.cat([L1, L2], dim=1))

        # cascade
        cas_w = torch.cat([L1, ref_fea_l[0]], dim=1)
        cas_w = self.cas_conv4(self.cas_conv3(self.cas_conv2(self.cas_conv1(cas_w))))
        cas_mask = self.cas_mask(cas_w).view(B, self.g, 1, self.cas_k ** 2, H, W)
        # generate mot similar feas
        cas_norm_ref_fea = F.normalize(ref_fea_l[0], dim=1)
        cas_norm_nbr_fea = F.normalize(L1, dim=1)
        cas_corr = self.L3_corr(cas_norm_ref_fea, cas_norm_nbr_fea).view(B, -1, H, W)
        _, cas_corr_ind = torch.topk(cas_corr, self.nbr, dim=1)
        cas_corr_ind = cas_corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        cas_ind_row_add = cas_corr_ind // self.patch_size[0] * (W + self.add_num[0])
        cas_ind_col_add = cas_corr_ind % self.patch_size[0]
        cas_corr_ind = cas_ind_row_add + cas_ind_col_add
        # generate top-left indexes
        cas_lt_ind = y * (W + self.add_num[0]) + x
        cas_lt_ind = cas_lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        cas_corr_ind = (cas_corr_ind + cas_lt_ind).view(-1)
        cas_nbr = F.unfold(L1, self.cor_k, dilation=1, padding=self.pad_size[0], stride=1)
        cas = cas_nbr[ind_B, :, cas_corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        # cas = cas.permute(0, 2, 1, 3, 4).contiguous().view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        cas = self.cas_nn_conv(cas)
        cas = cas.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        cas = cas.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        cas = self.relu((cas * cas_mask).sum(dim=3).view(B, C, H, W))

        return cas


class TemporalFusion(nn.Module):
    def __init__(self, nf, n_frame):
        super(TemporalFusion, self).__init__()
        self.n_frame = n_frame

        self.ref_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.nbr_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.up_conv = nn.Conv2d(nf * n_frame, nf * 4, 1, 1, bias=True)
        self.ps = nn.PixelShuffle(2)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()

        emb_ref = self.ref_conv(x[:, N // 2, :, :, :].clone())
        emb = self.nbr_conv(x.view(-1, C, H, W)).view(B, N, C, H, W)

        cor_l = []
        for i in range(N):
            cor = torch.sum(emb[:, i, :, :, :] * emb_ref, dim=1, keepdim=True)
            cor_l.append(cor)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aggr_fea = x.view(B, -1, H, W) * cor_prob

        fea = self.lrelu(self.up_conv(aggr_fea))
        out = self.ps(fea)

        return out


class TMCAM(nn.Module):
    def __init__(self, nf, n_frame, nbr, n_group, kernels, patches, cor_ksize):
        super(TMCAM, self).__init__()
        self.n_frame = n_frame

        self.aggr = Aggregate(nf=nf, nbr=nbr, n_group=n_group, kernels=kernels, patches=patches, cor_ksize=cor_ksize)
        self.tf = TemporalFusion(nf=nf, n_frame=n_frame)

    def forward(self, x):
        L1_fea, L2_fea, L3_fea = x
        center = self.n_frame // 2

        ref_fea_l = [
            L1_fea[:, center, :, :, :].clone(), L2_fea[:, center, :, :, :].clone(),
            L3_fea[:, center, :, :, :].clone()
        ]

        aggr_fea_l = []
        for i in range(self.n_frame):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aggr_fea_l.append(self.aggr(nbr_fea_l, ref_fea_l))

        aggr_fea = torch.stack(aggr_fea_l, dim=1)  # [B, N, C, H, W]
        out = self.tf(aggr_fea)

        return out


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        in_chl = config.MODEL.IN_CHANNEL
        out_chl = config.MODEL.OUT_CHANNEL
        nf = config.MODEL.N_CHANNEL
        front_blk = config.MODEL.FRONT_BLOCK
        n_frame = config.MODEL.N_FRAME
        nbr = config.MODEL.NEAREST_NEIGHBOR
        n_group = config.MODEL.N_GROUP
        kernels = config.MODEL.KERNELS
        patches = config.MODEL.PATCHES
        cor_ksize = config.MODEL.CORRELATION_KERNEL
        back_blk = config.MODEL.BACK_BLOCK
        n_level = config.MODEL.N_LEVEL

        self.first_conv = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        block = functools.partial(ResidualBlock_noBN_noAct, nf=nf)
        self.fea_ext = make_layer(block, front_blk)

        self.L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.tmcam = TMCAM(nf=nf, n_frame=n_frame, nbr=nbr, n_group=n_group, kernels=kernels, patches=patches,
                           cor_ksize=cor_ksize)
        self.cncam = CNCAM(nf=nf, n_level=n_level)

        self.recon = make_layer(block, back_blk)
        self.up_conv = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.ps = nn.PixelShuffle(2)
        self.hr_conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.last_conv = nn.Conv2d(64, out_chl, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.criterion = CharbonnierLoss(mode='mean')

    def forward(self, x, gt=None):
        B, N, C, H, W = x.size()
        x_center = x[:, N // 2, :, :, :].contiguous()

        L1_fea = self.lrelu(self.first_conv(x.view(-1, C, H, W)))
        L1_fea = self.fea_ext(L1_fea)

        L2_fea = self.lrelu(self.L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.L2_conv2(L2_fea))

        L3_fea = self.lrelu(self.L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea = self.tmcam([L1_fea, L2_fea, L3_fea])
        fea = self.cncam(fea)

        fea = self.recon(fea)
        fea = self.lrelu(self.ps(self.up_conv(fea)))
        fea = self.lrelu(self.hr_conv(fea))
        residual = self.last_conv(fea)

        bic = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out = bic + residual

        if gt is not None:
            loss_dict = dict(CB=self.criterion(out, gt))
            return loss_dict
        else:
            return out


if __name__ == '__main__':
    from utils import model_opr
    from config import config

    net = Network(config)
    model_opr.load_model(net, 'mucan_vimeo.pth', strict=True, cpu=True)
    
    device = torch.device('cuda')
    net = net.to(device)

    dir_path = '~/vincent/datasets/Vimeo-90K/vimeo_septuplet/sequences_matlabLRx4/00001/1000'


    sys.exit()

    import json
    saved = json.load(open('saved_parameters.json', 'r'))
    new = json.load(open('mapping.json', 'r'))

    mapping = dict()
    for i, name in enumerate(saved[0]):
        mapping[name] = new[0][i]

    new_state_dict = dict()

    state_dict = torch.load('200000_G.pth')
    for name, p in state_dict.items():
        new_state_dict[mapping[name]] = p

    torch.save(new_state_dict, 'mucan_vimeo.pth')
    sys.exit()



    import json
    saved = json.load(open('saved_parameters.json', 'r'))

    state_dict = torch.load('200000_G.pth')
    details = [[], []]
    for name, p in state_dict.items():
        details[0].append(name)
        details[1].append(list(p.size()))

    for i in range(len(details[0])):
        if saved[0][i] != details[0][i]:
            print(saved[0][i], details[0][i])
    # with open('saved_parameters.json', 'w') as f:
    #     json.dump(details, f, indent=4)
    sys.exit()

    device = torch.device('cuda')
    net = Network(nf=128, back_blk=40)
    net.to(device)
    cnt = 0
    for p in net.parameters():
        cnt += p.numel()
    print(cnt)
    input = torch.randn((1, 5, 3, 64, 64)).to(device)
    with torch.no_grad():
        out = net(input)

    details = [[], []]
    for name, p in net.named_parameters():
        details[0].append(name)
        details[1].append(list(p.size()))
    with open('parameters.json', 'w') as f:
        json.dump(details, f, indent=4)


