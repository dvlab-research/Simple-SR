import argparse
import cv2
import numpy as np
import os
import sys

import torch

from utils.model_opr import load_model
from utils.common import tensor2img


def get_network(model_path):
    if 'REDS' in model_path:
        from exps.MuCAN_REDS.config import config
        from exps.MuCAN_REDS.network import Network
    elif 'Vimeo' in model_path:
        from exps.MuCAN_Vimeo90K.config import config
        from exps.MuCAN_Vimeo90K.network import Network
    else:
        raiseNotImplementedError("Illenal model: not implemented!")

    return config, Network(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    config, model = get_network(args.model_path) 
    device = torch.device('cuda')
    model = model.to(device)
    load_model(model, args.model_path, strict=True)
    
    ipath_l = []
    for f in sorted(os.listdir(args.input_path)):
        if f.endswith('png') or f.endswith('jpg'):
            ipath_l.append(os.path.join(args.input_path, f))
    num_img = len(ipath_l)

    down = config.MODEL.DOWN
    scale = config.MODEL.SCALE
    half_n = config.MODEL.N_FRAME // 2
    with torch.no_grad():
        for i, f in enumerate(ipath_l):
            img_name = f.split('/')[-1]
            print(img_name)
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
            output_path = os.path.join(args.output_path, img_name)
            cv2.imwrite(output_path, output)

    print('Finished!')

