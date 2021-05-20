from functools import partial
import math
import numpy as np
import random

import torch

from utils import samplers


def worker_init_fn_seed(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_loader(dataset, config, rank, seed=None, is_dist=True, is_shuffle=True, start_iter=0):
    if is_dist:
        sampler = samplers.DistributedSampler(dataset, shuffle=is_shuffle)
    elif is_shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, config.DATALOADER.IMG_PER_GPU, drop_last=False)
    batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, config.SOLVER.MAX_ITER, start_iter)

    if seed is not None:
        worker_init_fn = partial(worker_init_fn_seed, num_workers=config.DATALOADER.NUM_WORKERS, rank=rank, seed=seed)
    else:
        worker_init_fn = None
    loader = torch.utils.data.DataLoader(dataset, num_workers=config.DATALOADER.NUM_WORKERS,
                                         batch_sampler=batch_sampler, worker_init_fn=worker_init_fn)

    return loader


def val_loader(dataset, config, local_rank, num_gpu):
    num_data = len(dataset)
    data_per_gpu = math.ceil(num_data / num_gpu)
    st = local_rank * data_per_gpu
    ed = min(num_data, st + data_per_gpu)
    indices = range(st, ed)
    subset = torch.utils.data.Subset(dataset, indices)

    sampler = torch.utils.data.sampler.SequentialSampler(subset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, config.VAL.IMG_PER_GPU, drop_last=False)

    loader = torch.utils.data.DataLoader(subset, num_workers=config.VAL.NUM_WORKERS, batch_sampler=batch_sampler)

    return loader

