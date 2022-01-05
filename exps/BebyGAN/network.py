import functools
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.modules.rrdb import RRDBNet
from utils.loss import AdversarialLoss, PerceptualLoss, BBL
from utils.modules.discriminator import Discriminator_VGG_192


class Generator(RRDBNet):
    def __init__(self, config):
        super(Generator, self).__init__(in_nc=config.MODEL.G.IN_CHANNEL,
                                        out_nc=config.MODEL.G.OUT_CHANNEL,
                                        nf=config.MODEL.G.N_CHANNEL,
                                        nb=config.MODEL.G.N_BLOCK,
                                        gc=config.MODEL.G.N_GROWTH_CHANNEL)


class Discriminator(Discriminator_VGG_192):
    def __init__(self, config):
        super(Discriminator, self).__init__(in_chl=config.MODEL.D.IN_CHANNEL,
                                            nf=config.MODEL.D.N_CHANNEL)


class Network:
    def __init__(self, config):
        self.G = Generator(config)
        self.D = Discriminator(config)

        self.recon_loss_weight = config.MODEL.BBL_WEIGHT
        self.adv_loss_weight = config.MODEL.ADV_LOSS_WEIGHT
        self.bp_loss_weight = config.MODEL.BACK_PROJECTION_LOSS_WEIGHT
        self.use_pcp = config.MODEL.USE_PCP_LOSS
        self.recon_criterion = BBL(alpha=config.MODEL.BBL_ALPHA,
                                   beta=config.MODEL.BBL_BETA,
                                   ksize=config.MODEL.BBL_KSIZE,
                                   pad=config.MODEL.BBL_PAD,
                                   stride=config.MODEL.BBL_STRIDE,
                                   criterion=config.MODEL.BBL_TYPE)
        self.adv_criterion = AdversarialLoss(gan_type=config.MODEL.D.LOSS_TYPE)
        self.bp_criterion = nn.L1Loss(reduction='mean')
        if self.use_pcp:
            self.pcp_criterion = PerceptualLoss(layer_weights=config.MODEL.VGG_LAYER_WEIGHTS,
                                                vgg_type=config.MODEL.VGG_TYPE,
                                                use_input_norm=config.MODEL.USE_INPUT_NORM,
                                                use_pcp_loss=config.MODEL.USE_PCP_LOSS,
                                                use_style_loss=config.MODEL.USE_STYLE_LOSS,
                                                norm_img=config.MODEL.NORM_IMG,
                                                criterion=config.MODEL.PCP_LOSS_TYPE)
            self.pcp_loss_weight = config.MODEL.PCP_LOSS_WEIGHT
            self.style_loss_weight = config.MODEL.STYLE_LOSS_WEIGHT

    def set_device(self, device):
        self.G = self.G.to(device)
        self.D = self.D.to(device)
        self.recon_criterion = self.recon_criterion.to(device)
        self.adv_criterion = self.adv_criterion.to(device)
        self.bp_criterion = self.bp_criterion.to(device)
        if self.use_pcp:
            self.pcp_criterion = self.pcp_criterion.to(device)


if __name__ == '__main__':
    from config import config

    net = Network(config)
    print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.G.parameters())/1e6))

