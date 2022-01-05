import cv2
import os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../..')

import torch

from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr


def validate(model, val_loader, config, device, iteration, save_path='.'):
    with torch.no_grad():
        psnr_l = []
        ssim_l = []

        for idx, (lr_img, hr_img) in enumerate(val_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            output = model.G(lr_img)

            output = tensor2img(output)
            gt = tensor2img(hr_img)

            if config.VAL.SAVE_IMG:
                ipath = os.path.join(save_path, '%d_%03d.png' % (iteration, idx))
                cv2.imwrite(ipath, np.concatenate([output, gt], axis=1))

            output = output.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0

            if config.VAL.TO_Y:
                output = bgr2ycbcr(output, only_y=True)
                gt = bgr2ycbcr(gt, only_y=True)

            if config.VAL.CROP_BORDER != 0:
                cb = config.VAL.CROP_BORDER
                output = output[cb:-cb, cb:-cb]
                gt = gt[cb:-cb, cb:-cb]

            psnr = calculate_psnr(output * 255, gt * 255)
            ssim = calculate_ssim(output * 255, gt * 255)
            psnr_l.append(psnr)
            ssim_l.append(ssim)

        avg_psnr = sum(psnr_l) / len(psnr_l)
        avg_ssim = sum(ssim_l) / len(ssim_l)

    return avg_psnr, avg_ssim


if __name__ == '__main__':
    from config import config
    from network import Network
    from dataset import get_dataset
    from utils import dataloader
    from utils.model_opr import load_model

    config.VAL.DATASETS = ['BSDS100']
    config.VAL.SAVE_IMG = True

    model = Network(config)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.G = model.G.to(device)

    model_path = 'log/models/600000_G.pth'
    load_model(model.G, model_path, cpu=True)

    val_dataset = get_dataset(config.VAL)
    val_loader = dataloader.val_loader(val_dataset, config, 0, 1)
    psnr, ssim = validate(model, val_loader, config, device, 0, save_path='gt')
    print('PSNR: %.4f, SSIM: %.4f' % (psnr, ssim))
