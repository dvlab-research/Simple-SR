import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BBL(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, ksize=3, pad=0, stride=3, dist_norm='l2', criterion='l1'):
        super(BBL, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ksize = ksize
        self.pad = pad
        self.stride = stride
        self.dist_norm = dist_norm

        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif criterion == 'l2':
            self.criterion = torch.nn.L2loss(reduction='mean')
        else:
            raise NotImplementedError('%s criterion has not been supported.' % criterion)

    def pairwise_distance(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a BxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        if y is None:
            dist = dist - torch.diag(dist.diag())

        return torch.clamp(dist, 0.0, np.inf)
    
    def batch_pairwise_distance(self, x, y=None):
        '''
        Input: x is a BxNxd matrix
               y is an optional BxMxd matirx
        Output: dist is a BxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
                if y is not given then use 'y=x'.
        i.e. dist[b,i,j] = ||x[b,i,:]-y[b,j,:]||^2
        '''
        B, N, d = x.size()
        if self.dist_norm == 'l1':
            x_norm = x.view(B, N, 1, d)
            if y is not None:
                y_norm = y.view(B, 1, -1, d)
            else:
                y_norm = x.view(B, 1, -1, d)
            dist = torch.abs(x_norm - y_norm).sum(dim=3)
        elif self.dist_norm == 'l2':
            x_norm = (x ** 2).sum(dim=2).view(B, N, 1)
            if y is not None:
                M = y.size(1)
                y_t = torch.transpose(y, 1, 2)
                y_norm = (y ** 2).sum(dim=2).view(B, 1, M)
            else:
                y_t = torch.transpose(x, 1, 2)
                y_norm = x_norm.view(B, 1, N)

            dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
            # Ensure diagonal is zero if x=y
            if y is None:
                dist = dist - torch.diag_embed(torch.diagonal(dist, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            dist = torch.clamp(dist, 0.0, np.inf)
            # dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf) / d)
        else:
            raise NotImplementedError('%s norm has not been supported.' % self.dist_norm)

        return dist

    def forward(self, x, gt):
        p1 = F.unfold(x, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        B, C, H = p1.size()
        p1 = p1.permute(0, 2, 1).contiguous() # [B, H, C]

        p2 = F.unfold(gt, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2 = p2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_2 = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners = False)
        p2_2 = F.unfold(gt_2, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_2 = p2_2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_4 = F.interpolate(gt, scale_factor=0.25, mode='bicubic',align_corners = False)
        p2_4 = F.unfold(gt_4, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_4 = p2_4.permute(0, 2, 1).contiguous() # [B, H, C]
        p2_cat = torch.cat([p2, p2_2, p2_4], 1)

        score1 = self.alpha * self.batch_pairwise_distance(p1, p2_cat)
        score = score1 + self.beta * self.batch_pairwise_distance(p2, p2_cat) # [B, H, H]

        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, C]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)

        return loss


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')

    bbl_func = BBL().to(device)

    query_patch = torch.randn(1, 3, 48, 48).to(device)
    gt_patch = torch.randn(1, 3, 48, 48).to(device)

    with torch.no_grad():
        result = bbl_func(query_patch, gt_patch)
    print('result', result)