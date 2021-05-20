import cv2
import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset


class MixDataset(Dataset):
    def __init__(self, hr_paths, lr_paths, config):
        self.hr_paths = hr_paths
        self.lr_paths = lr_paths
        self.phase = config.PHASE
        self.input_width, self.input_height = config.INPUT_WIDTH, config.INPUT_HEIGHT
        self.scale = config.SCALE
        self.repeat = config.REPEAT
        self.value_range = config.VALUE_RANGE

        self._load_data()

    def _load_data(self):
        assert len(self.lr_paths) == len(self.hr_paths), 'Illegal hr-lr dataset mappings.'

        self.hr_list = []
        self.lr_list = []
        for hr_path in self.hr_paths:
            hr_imgs = sorted(os.listdir(hr_path))
            for hr_img in hr_imgs:
                self.hr_list.append(os.path.join(hr_path, hr_img))
        for lr_path in self.lr_paths:
            lr_imgs = sorted(os.listdir(lr_path))
            for lr_img in lr_imgs:
                self.lr_list.append(os.path.join(lr_path, lr_img))

        assert len(self.hr_list) == len(self.lr_list), 'Illegal hr-lr mappings.'

        self.data_len = len(self.hr_list)
        self.full_len = self.data_len * self.repeat

    def __len__(self):
        return self.full_len

    def __getitem__(self, index):
        idx = index % self.data_len

        url_hr = self.hr_list[idx]
        url_lr = self.lr_list[idx]

        img_hr = cv2.imread(url_hr, cv2.IMREAD_COLOR)
        img_lr = cv2.imread(url_lr, cv2.IMREAD_COLOR)

        if self.phase == 'train':
            h, w = img_lr.shape[:2]
            s = self.scale

            # random cropping
            y = random.randint(0, h - self.input_height)
            x = random.randint(0, w - self.input_width)
            img_lr = img_lr[y: y + self.input_height, x: x + self.input_width, :]
            img_hr = img_hr[y * s: (y + self.input_height) * s,
                            x * s: (x + self.input_width) * s, :]

            # horizontal flip
            if random.random() > 0.5:
                cv2.flip(img_lr, 1, img_lr)
                cv2.flip(img_hr, 1, img_hr)
            # vertical flip
            if random.random() > 0.5:
                cv2.flip(img_lr, 0, img_lr)
                cv2.flip(img_hr, 0, img_hr)
            # rotation 90 degree
            if random.random() > 0.5:
                img_lr = img_lr.transpose(1, 0, 2)
                img_hr = img_hr.transpose(1, 0, 2)

        # BGR to RGB, HWC to CHW, uint8 to float32
        img_lr = np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1)).astype(np.float32)
        img_hr = np.transpose(img_hr[:, :, [2, 1, 0]], (2, 0, 1)).astype(np.float32)

        # numpy array to tensor, [0, 255] to [0, 1]
        img_lr = torch.from_numpy(img_lr).float() / self.value_range
        img_hr = torch.from_numpy(img_hr).float() / self.value_range

        return img_lr, img_hr


if __name__ == '__main__':
    from easydict import EasyDict as edict
    config = edict()
    config.PHASE = 'train'
    config.INPUT_WIDTH = config.INPUT_HEIGHT = 64
    config.SCALE = 4
    config.REPEAT = 1
    config.VALUE_RANGE = 255.0

    D = MixDataset(hr_paths=['/data/liwenbo/datasets/DIV2K/DIV2K_train_HR_sub'],
                   lr_paths=['/data/liwenbo/datasets/DIV2K/DIV2K_train_LR_bicubic_sub/X4'],
                   config=config)
    print(D.data_len, D.full_len)
    lr, hr = D.__getitem__(5)
    print(lr.size(), hr.size())
    print('Done')
