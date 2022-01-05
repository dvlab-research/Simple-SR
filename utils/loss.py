import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.modules.vggNet import VGGFeatureExtractor


class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, out, gt):
        grad_out_x = out[:, :, :, 1:] - out[:, :, :, :-1]
        grad_out_y = out[:, :, 1:, :] - out[:, :, :-1, :]

        grad_gt_x = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        grad_gt_y = gt[:, :, 1:, :] - gt[:, :, :-1, :]

        loss_x = self.l1(grad_out_x, grad_gt_x)
        loss_y = self.l1(grad_out_y, grad_gt_y)

        loss = self.weight * (loss_x + loss_y)

        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, mode=None):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.mode = mode

    def forward(self, x, y, mask=None):
        N = x.size(1)
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        if mask is not None:
            loss = loss * mask
        if self.mode == 'sum':
            loss = torch.sum(loss) / N
        else:
            loss = loss.mean()
        return loss


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


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
            Default: False.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 use_pcp_loss=True,
                 use_style_loss=True,
                 norm_img=False,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.norm_img = norm_img
        self.use_pcp_loss = use_pcp_loss
        self.use_style_loss = use_style_loss
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError('%s criterion has not been supported.' % self.criterion_type)

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5

        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.use_pcp_loss:
            percep_loss = 0
            non_local_loss = None
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
                    # non_local_loss += self.non_local_criterion(x_features[k], gt_features[k]) * self.layer_weights[k] * 0.1
        else:
            percep_loss = None

        # calculate style loss
        if self.use_style_loss:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) -
                        self._gram_mat(gt_features[k]),
                        p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) \
                                  * self.layer_weights[k]
        else:
            style_loss = None

        return percep_loss, style_loss,non_local_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class AdversarialLoss(nn.Module):
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        """
        Args:
            gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
            real_label_val (float): The value for real label. Default: 1.0.
            fake_label_val (float): The value for fake label. Default: 0.0.
        """

        super(AdversarialLoss, self).__init__()

        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError('GAN type %s is not implemented.' % self.gan_type)

    def _wgan_loss(self, input, target):
        """
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise, return Tensor.
        """
        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        return loss

