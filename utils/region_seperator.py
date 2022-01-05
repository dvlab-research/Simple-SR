import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_flat_mask(img, kernel_size=7, std_thresh=0.03, scale=1):
    img = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
    B, _, H, W = img.size()
    r, g, b = torch.unbind(img, dim=1)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=1)
    l_img_pad = F.pad(l_img, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
    unf_img = F.unfold(l_img_pad, kernel_size=kernel_size, padding=0, stride=1)
    std_map = torch.std(unf_img, dim=1, keepdim=True).view(B, 1, H, W)
    mask = torch.lt(std_map, std_thresh).float()

    return mask
