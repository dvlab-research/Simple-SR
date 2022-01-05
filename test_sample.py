import argparse
import cv2
import numpy as np
import os
import sys

import torch

from utils.model_opr import load_model
from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr


def get_network(model_path):
    if 'REDS' in model_path:
        from exps.MuCAN_REDS.config import config
        from exps.MuCAN_REDS.network import Network
    elif 'Vimeo' in model_path:
        from exps.MuCAN_Vimeo90K.config import config
        from exps.MuCAN_Vimeo90K.network import Network
    elif 'LAPAR_A_x2' in model_path:
        from exps.LAPAR_A_x2.config import config
        from exps.LAPAR_A_x2.network import Network
    elif 'LAPAR_A_x3' in model_path:
        from exps.LAPAR_A_x3.config import config
        from exps.LAPAR_A_x3.network import Network
    elif 'LAPAR_A_x4' in model_path:
        from exps.LAPAR_A_x4.config import config
        from exps.LAPAR_A_x4.network import Network
    elif 'LAPAR_B_x2' in model_path:
        from exps.LAPAR_B_x2.config import config
        from exps.LAPAR_B_x2.network import Network
    elif 'LAPAR_B_x3' in model_path:
        from exps.LAPAR_B_x3.config import config
        from exps.LAPAR_B_x3.network import Network
    elif 'LAPAR_B_x4' in model_path:
        from exps.LAPAR_B_x4.config import config
        from exps.LAPAR_B_x4.network import Network
    elif 'LAPAR_C_x2' in model_path:
        from exps.LAPAR_C_x2.config import config
        from exps.LAPAR_C_x2.network import Network
    elif 'LAPAR_C_x3' in model_path:
        from exps.LAPAR_C_x3.config import config
        from exps.LAPAR_C_x3.network import Network
    elif 'LAPAR_C_x4' in model_path:
        from exps.LAPAR_C_x4.config import config
        from exps.LAPAR_C_x4.network import Network
    elif 'BebyGAN_x4' in model_path:
        from exps.BebyGAN.config import config
        from exps.BebyGAN.network import Network
    else:
        print('Illenal model: not implemented!')
        sys.exit(1)

    # an ugly operation
    if 'KERNEL_PATH' in config.MODEL:
        config.MODEL.KERNEL_PATH = config.MODEL.KERNEL_PATH.replace('../', '')

    if 'BebyGAN' in model_path:
        return config, Network(config).G

    return config, Network(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr_type', type=str, default='SISR')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--gt_path', type=str, default=None)
    args = parser.parse_args()

    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    print('Loading Network ...')
    config, model = get_network(args.model_path)
    device = torch.device('cuda')
    model = model.to(device)
    load_model(model, args.model_path, strict=True)

    down = config.MODEL.DOWN
    scale = config.MODEL.SCALE

    print('Reading Images ...')
    ipath_l = []
    for f in sorted(os.listdir(args.input_path)):
        if f.endswith('png') or f.endswith('jpg'):
            ipath_l.append(os.path.join(args.input_path, f))

    if args.gt_path:
        gpath_l = []
        for f in sorted(os.listdir(args.gt_path)):
            if f.endswith('png') or f.endswith('jpg'):
                gpath_l.append(os.path.join(args.gt_path, f))
        psnr_l = []
        ssim_l = []

    if args.sr_type == 'SISR':
        with torch.no_grad():
            for i, f in enumerate(ipath_l):
                img_name = f.split('/')[-1]
                print('Processing: %s' % img_name)
                lr_img = cv2.imread(f, cv2.IMREAD_COLOR)
                lr_img = np.transpose(lr_img[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.0
                lr_img = torch.from_numpy(lr_img).float().to(device).unsqueeze(0)

                _, C, H, W = lr_img.size()

                need_pad = False
                if H % down != 0 or W % down != 0:
                    need_pad = True
                    pad_y_t = (down - H % down) % down // 2
                    pad_y_b = (down - H % down) % down - pad_y_t
                    pad_x_l = (down - W % down) % down // 2
                    pad_x_r = (down - W % down) % down - pad_x_l
                    lr_img = torch.nn.functional.pad(lr_img, pad=(pad_x_l, pad_x_r, pad_y_t, pad_y_b), mode='replicate')

                output = model(lr_img)

                if need_pad:
                    y_end = -pad_y_b * scale if pad_y_b != 0 else output.size(2)
                    x_end = -pad_x_r * scale if pad_x_r != 0 else output.size(3)
                    output = output[:, :, pad_y_t * scale: y_end, pad_x_l * scale: x_end]

                output = tensor2img(output)
                if args.output_path:
                    output_path = os.path.join(args.output_path, img_name)
                    cv2.imwrite(output_path, output)

                if args.gt_path:
                    output = output.astype(np.float32) / 255.0
                    gt = cv2.imread(gpath_l[i], cv2.IMREAD_COLOR).astype(np.float32) / 255.0

                    # to y channel
                    output = bgr2ycbcr(output, only_y=True)
                    gt = bgr2ycbcr(gt, only_y=True)

                    output = output[scale:-scale, scale:-scale]
                    gt = gt[scale:-scale, scale:-scale]

                    psnr = calculate_psnr(output * 255, gt * 255)
                    ssim = calculate_ssim(output * 255, gt * 255)

                    psnr_l.append(psnr)
                    ssim_l.append(ssim)

    elif args.sr_type == 'VSR':
        num_img = len(ipath_l)

        half_n = config.MODEL.N_FRAME // 2
        with torch.no_grad():
            for i, f in enumerate(ipath_l):
                img_name = f.split('/')[-1]
                print('Processing: %s' % img_name)
                nbr_l = []
                for j in range(i - half_n, i + half_n + 1):
                    if j < 0:
                        ipath = ipath_l[i + half_n - j]
                    elif j >= num_img:
                        ipath = ipath_l[i - half_n - (j - num_img + 1)]
                    else:
                        ipath = ipath_l[j]
                    nbr_img = cv2.imread(ipath, cv2.IMREAD_COLOR)
                    nbr_l.append(nbr_img)
                lr_imgs = np.stack(nbr_l, axis=0)
                lr_imgs = np.transpose(lr_imgs[:, :, :, ::-1], (0, 3, 1, 2)).astype(np.float32) / 255.0
                lr_imgs = torch.from_numpy(lr_imgs).float().to(device)

                N, C, H, W = lr_imgs.size()

                need_pad = False
                if H % down != 0 or W % down != 0:
                    need_pad = True
                    pad_y_t = (down - H % down) % down // 2
                    pad_y_b = (down - H % down) % down - pad_y_t
                    pad_x_l = (down - W % down) % down // 2
                    pad_x_r = (down - W % down) % down - pad_x_l
                    lr_imgs = torch.nn.functional.pad(lr_imgs, pad=(pad_x_l, pad_x_r, pad_y_t, pad_y_b), mode='replicate')
                lr_imgs = lr_imgs.unsqueeze(0)

                output = model(lr_imgs)

                if need_pad:
                    y_end = -pad_y_b * scale if pad_y_b != 0 else output.size(2)
                    x_end = -pad_x_r * scale if pad_x_r != 0 else output.size(3)
                    output = output[:, :, pad_y_t * scale: y_end, pad_x_l * scale: x_end]

                output = tensor2img(output)
                if args.output_path:
                    output_path = os.path.join(args.output_path, img_name)
                    cv2.imwrite(output_path, output)

                if args.gt_path:
                    output = output.astype(np.float32) / 255.0
                    gt = cv2.imread(gpath_l[i], cv2.IMREAD_COLOR).astype(np.float32) / 255.0

                    # to y channel
                    output = bgr2ycbcr(output, only_y=True)
                    gt = bgr2ycbcr(gt, only_y=True)

                    output = output[scale:-scale, scale:-scale]
                    gt = gt[scale:-scale, scale:-scale]

                    psnr = calculate_psnr(output * 255, gt * 255)
                    ssim = calculate_ssim(output * 255, gt * 255)

                    psnr_l.append(psnr)
                    ssim_l.append(ssim)

    else:
        print('Illenal SR type: not implemented!')
        sys.exit(1)

    if args.gt_path:
        avg_psnr = sum(psnr_l) / len(psnr_l)
        avg_ssim = sum(ssim_l) / len(ssim_l)
        print('--------- Result ---------')
        print('PSNR: %.2f, SSIM:%.4f' % (avg_psnr, avg_ssim))

    print('Finished!')

