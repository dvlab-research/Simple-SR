import math

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


def make_optimizer(config, model, num_gpu=None):
    if num_gpu is None:
        lr = config.SOLVER.BASE_LR
    else:
        lr = config.SOLVER.BASE_LR * num_gpu

    if config.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=config.SOLVER.WEIGHT_DECAY)
    elif config.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=config.SOLVER.MOMENTUM,
                              weight_decay=config.SOLVER.WEIGHT_DECAY)
    else:
        raise ValueError('Illegal optimizer.')

    return optimizer


def make_optimizer_sep(model, base_lr, type='Adam', beta1=0.9, beta2=0.999, weight_decay=0, momentum=0, num_gpu=None):
    if num_gpu is None:
        lr = base_lr
    else:
        lr = base_lr * num_gpu

    if type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)
    elif type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif type == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, betas=(beta1, beta2), eps=1e-8, weight_decay=0.01)
    elif type == 'Adamax':
        optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)
    else:
        raise ValueError('Illegal optimizer.')

    return optimizer


def make_lr_scheduler(config, optimizer):
    w_iter = config.SOLVER.WARM_UP_ITER
    w_fac = config.SOLVER.WARM_UP_FACTOR
    max_iter = config.SOLVER.MAX_ITER
    lr_lambda = lambda iteration: w_fac + (1 - w_fac) * iteration / w_iter \
        if iteration < w_iter \
        else 1 / 2 * (1 + math.cos((iteration - w_iter) / (max_iter - w_iter) * math.pi))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return scheduler


class CosineAnnealingLR_warmup(_LRScheduler):
    def __init__(self, config, optimizer, base_lr, last_epoch=-1, min_lr=1e-7):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.w_iter = config.SOLVER.WARM_UP_ITER
        self.w_fac = config.SOLVER.WARM_UP_FACTOR
        self.T_period = config.SOLVER.T_PERIOD
        self.last_restart = 0
        self.T_max = self.T_period[0]
        assert config.SOLVER.MAX_ITER == self.T_period[-1], 'Illegal training period setting.'
        super(CosineAnnealingLR_warmup, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch - self.last_restart < self.w_iter:
            ratio = self.w_fac + (1 - self.w_fac) * (self.last_epoch - self.last_restart) / self.w_iter
            return [(self.base_lr - self.min_lr) * ratio + self.min_lr for group in self.optimizer.param_groups]
        elif self.last_epoch in self.T_period:
            self.last_restart = self.last_epoch
            if self.last_epoch != self.T_period[-1]:
                self.T_max = self.T_period[self.T_period.index(self.last_epoch) + 1]
            return [self.min_lr for group in self.optimizer.param_groups]
        else:
            ratio = 1 / 2 * (1 + math.cos(
                (self.last_epoch - self.last_restart - self.w_iter) / (self.T_max - self.last_restart - self.w_iter) * math.pi))
            return [(self.base_lr - self.min_lr) * ratio + self.min_lr for group in self.optimizer.param_groups]
