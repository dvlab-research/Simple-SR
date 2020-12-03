from easydict import EasyDict as edict


class Config:
    # dataset

    # dataloader

    # model
    MODEL = edict()
    MODEL.SCALE = 4
    MODEL.KERNEL_SIZE = 5
    MODEL.KERNEL_PATH = 'kernel/gaussian_k5_dog_v3.pkl'
    MODEL.IN_CHANNEL = 3
    MODEL.N_CHANNEL = 32
    MODEL.RES_BLOCK = 4
    MODEL.N_WEIGHT = 72
    MODEL.DOWN = 1
    MODEL.DEVICE = 'cuda'

    # solver

    # initialization

    # log and save

    # validation


config = Config()



