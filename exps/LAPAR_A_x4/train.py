import argparse
import os
import os.path as osp
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../..')
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from config import config
from utils import common, dataloader, solver, model_opr
from dataset import get_dataset
from network import Network
from validate import validate


def init_dist(local_rank):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # initialization
    rank = 0
    num_gpu = 1
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        num_gpu = int(os.environ['WORLD_SIZE'])
        distributed = num_gpu > 1
    if distributed:
        rank = args.local_rank
        init_dist(rank)
    common.init_random_seed(config.DATASET.SEED + rank)

    # set up dirs and log
    exp_dir, cur_dir = osp.split(osp.split(osp.realpath(__file__))[0])
    root_dir = osp.split(exp_dir)[0]
    log_dir = osp.join(root_dir, 'logs', cur_dir)
    model_dir = osp.join(log_dir, 'models')
    solver_dir = osp.join(log_dir, 'solvers')
    if rank <= 0:
        common.mkdir(log_dir)
        ln_log_dir = osp.join(exp_dir, cur_dir, 'log')
        if not osp.exists(ln_log_dir):
            os.system('ln -s %s log' % log_dir)
        common.mkdir(model_dir)
        common.mkdir(solver_dir)
        save_dir = osp.join(log_dir, 'saved_imgs')
        common.mkdir(save_dir)
        tb_dir = osp.join(log_dir, 'tb_log')
        tb_writer = SummaryWriter(tb_dir)
        common.setup_logger('base', log_dir, 'train', level=logging.INFO, screen=True, to_file=True)
        logger = logging.getLogger('base')

    # dataset
    train_dataset = get_dataset(config.DATASET)
    train_loader = dataloader.train_loader(train_dataset, config, rank=rank, seed=config.DATASET.SEED,
                                           is_dist=distributed)
    if rank <= 0:
        val_dataset = get_dataset(config.VAL)
        val_loader = dataloader.val_loader(val_dataset, config, rank, 1)
        data_len = val_dataset.data_len

    # model
    model = Network(config)
    if rank <= 0:
        print(model)

    if config.CONTINUE_ITER:
        model_path = osp.join(model_dir, '%d.pth' % config.CONTINUE_ITER)
        if rank <= 0:
            logger.info('[Continue] Iter: %d' % config.CONTINUE_ITER)
        model_opr.load_model(model, model_path, strict=True, cpu=True)
    elif config.INIT_MODEL:
        if rank <= 0:
            logger.info('[Initialize] Model: %s' % config.INIT_MODEL)
        model_opr.load_model(model, config.INIT_MODEL, strict=True, cpu=True)

    device = torch.device(config.MODEL.DEVICE)
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    # solvers
    optimizer = solver.make_optimizer(config, model)  # lr without X num_gpu
    lr_scheduler = solver.CosineAnnealingLR_warmup(config, optimizer, config.SOLVER.BASE_LR)
    iteration = 0

    if config.CONTINUE_ITER:
        solver_path = osp.join(solver_dir, '%d.solver' % config.CONTINUE_ITER)
        iteration = model_opr.load_solver(optimizer, lr_scheduler, solver_path)

    max_iter = max_psnr = max_ssim = 0
    for lr_img, hr_img in train_loader:
        model.train()
        iteration = iteration + 1

        optimizer.zero_grad()

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        loss_dict = model(lr_img, gt=hr_img)
        total_loss = sum(loss for loss in loss_dict.values())
        total_loss.backward()

        optimizer.step()
        lr_scheduler.step()

        if rank <= 0:
            if iteration % config.LOG_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                log_str = 'Iter: %d, LR: %.3e, ' % (iteration, optimizer.param_groups[0]['lr'])
                for key in loss_dict:
                    tb_writer.add_scalar(key, loss_dict[key].mean(), global_step=iteration)
                    log_str += key + ': %.4f, ' % float(loss_dict[key])
                logger.info(log_str)

            if iteration % config.SAVE_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                logger.info('[Saving] Iter: %d' % iteration)
                model_path = osp.join(model_dir, '%d.pth' % iteration)
                solver_path = osp.join(solver_dir, '%d.solver' % iteration)
                model_opr.save_model(model, model_path)
                model_opr.save_solver(optimizer, lr_scheduler, iteration, solver_path)

            if iteration % config.VAL.PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                logger.info('[Validating] Iter: %d' % iteration)
                model.eval()
                with torch.no_grad():
                    psnr, ssim = validate(model, val_loader, config, device, iteration, save_path=save_dir)
                if psnr > max_psnr:
                    max_psnr, max_ssim, max_iter = psnr, ssim, iteration
                logger.info('[Val Result] Iter: %d, PSNR: %.4f, SSIM: %.4f' % (iteration, psnr, ssim))
                logger.info('[Best Result] Iter: %d, PSNR: %.4f, SSIM: %.4f' % (max_iter, max_psnr, max_ssim))

        if iteration >= config.SOLVER.MAX_ITER:
            break

    if rank <= 0:
        logger.info('Finish training process!')
        logger.info('[Final Best Result] Iter: %d, PSNR: %.4f, SSIM: %.4f' % (max_iter, max_psnr, max_ssim))


if __name__ == '__main__':
    main()
