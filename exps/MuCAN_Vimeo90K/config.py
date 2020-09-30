from easydict import EasyDict as edict
import numpy as np


class Config:
    # model
    MODEL = edict()
    MODEL.N_FRAME = 7
    MODEL.SCALE = 4
    MODEL.IN_CHANNEL = 3
    MODEL.OUT_CHANNEL = 3
    MODEL.N_CHANNEL = 128
    MODEL.FRONT_BLOCK = 5
    MODEL.NEAREST_NEIGHBOR = 4
    MODEL.N_GROUP = 8
    MODEL.KERNELS = [3, 3, 3, 3]
    MODEL.PATCHES = [7, 11, 15]
    MODEL.CORRELATION_KERNEL = 3
    MODEL.BACK_BLOCK = 20
    MODEL.N_LEVEL= 3
    MODEL.DOWN = 4
    MODEL.DEVICE = 'cuda'


config = Config()



