from easydict import EasyDict as edict


class Config:
    # dataset
    DATASET = edict()
    DATASET.TYPE = 'MixDataset'
    DATASET.DATASETS = ['DIV2K', 'Flickr2K']
    DATASET.SPLITS = ['TRAIN', 'TRAIN']
    DATASET.PHASE = 'train'
    DATASET.INPUT_HEIGHT = 64
    DATASET.INPUT_WIDTH = 64
    DATASET.SCALE = 4
    DATASET.REPEAT = 1
    DATASET.VALUE_RANGE = 255.0
    DATASET.SEED = 100

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 32
    DATALOADER.NUM_WORKERS = 8

    # model
    MODEL = edict()
    MODEL.SCALE = DATASET.SCALE
    MODEL.KERNEL_SIZE = 5
    MODEL.KERNEL_PATH = '../../kernel/kernel_72_k5.pkl'
    MODEL.IN_CHANNEL = 3
    MODEL.N_CHANNEL = 16
    MODEL.RES_BLOCK = 2
    MODEL.N_WEIGHT = 72
    MODEL.DOWN = 1
    MODEL.DEVICE = 'cuda'

    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.BASE_LR = 4e-4
    SOLVER.BETA1 = 0.9
    SOLVER.BETA2 = 0.999
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.T_PERIOD = [200000, 400000, 600000]
    SOLVER.MAX_ITER = SOLVER.T_PERIOD[-1]

    # initialization
    CONTINUE_ITER = None
    INIT_MODEL = None

    # log and save
    LOG_PERIOD = 20
    SAVE_PERIOD = 10000

    # validation
    VAL = edict()
    VAL.PERIOD = 10000
    VAL.TYPE = 'MixDataset'
    VAL.DATASETS = ['BSDS100']
    VAL.SPLITS = ['VAL']
    VAL.PHASE = 'val'
    VAL.INPUT_HEIGHT = None
    VAL.INPUT_WIDTH = None
    VAL.SCALE = DATASET.SCALE
    VAL.REPEAT = 1
    VAL.VALUE_RANGE = 255.0
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.SAVE_IMG = False
    VAL.TO_Y = True
    VAL.CROP_BORDER = VAL.SCALE


config = Config()



