import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def load_model(model, model_path, strict=True, cpu=False):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    if cpu:
        loaded_model = torch.load(model_path, map_location='cpu')
    else:
        loaded_model = torch.load(model_path)
    model.load_state_dict(loaded_model, strict=strict)


def load_solver(optimizer, lr_scheduler, solver_path):
    loaded_solver = torch.load(solver_path)
    loaded_optimizer = loaded_solver['optimizer']
    loaded_lr_scheduler = loaded_solver['lr_scheduler']
    iteration = loaded_solver['iteration']
    optimizer.load_state_dict(loaded_optimizer)
    lr_scheduler.load_state_dict(loaded_lr_scheduler)

    return iteration


def save_model(model, model_path):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    torch.save(model.state_dict(), model_path)


def save_solver(optimizer, lr_scheduler, iteration, solver_path):
    solver = dict()
    solver['optimizer'] = optimizer.state_dict()
    solver['lr_scheduler'] = lr_scheduler.state_dict()
    solver['iteration'] = iteration
    torch.save(solver, solver_path)
