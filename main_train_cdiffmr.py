'''
# -----------------------------------------
Main Program for Training
CDiff for MRI_Recon
by XXX
# -----------------------------------------
'''

import os
import sys
import math
import argparse
import random
import cv2
import numpy as np
import logging
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_early_stopping

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from tensorboardX import SummaryWriter
from collections import OrderedDict
from skimage.transform import resize
import lpips
import wandb


def main(json_path=''):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    # opt['dist'] = parser.parse_args().dist

    # distributed settings
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # update opt
    init_iter_model_DM, init_path_model_DM = option.find_last_checkpoint(opt['path']['models'], net_type='model_DM')
    init_iter_model_EMA, init_path_model_EMA = option.find_last_checkpoint(opt['path']['models'], net_type='model_EMA')
    init_iter_optimizer_DM, init_path_optimizer_DM = option.find_last_checkpoint(opt['path']['models'], net_type='optimizer_DM')
    current_step = max(init_iter_model_DM, init_iter_model_EMA, init_iter_optimizer_DM)

    if not opt["use_pretrain_weight"]:
        opt['path']['pretrained_model_DM'] = init_path_model_DM
        opt['path']['pretrained_model_EMA'] = init_path_model_EMA
        opt['path']['pretrained_optimizer_DM'] = init_path_optimizer_DM

    # save opt to  a '../option.json' file
    if opt['rank'] == 0:
        option.save(opt)

    # return None for missing key
    opt = option.dict_to_nonedict(opt)

    # Do not support DDP when using WANDB Sweep
    if opt['wandb']['is_sweep']:
        assert opt['rank'] == 0, 'Do not support DDP when using WANDB Sweep'

    # configure logger
    if opt['rank'] == 0:

        # wandb init
        os.environ['WANDB_MODE'] = opt['wandb']['mode']
        wandb.init(project=opt['wandb']['project_name'], entity="XXX")

        # sweep parameter
        # check here when changing sweep yaml
        if opt['wandb']['is_sweep']:
            pass
            # opt['train']['model_DM_optimizer_lr'] = wandb.config.model_DM_optimizer_lr
            # opt['train']['model_DM_optimizer_type'] = wandb.config.model_DM_optimizer_type
            # opt['datasets']['train']['dataloader_batch_size'] = wandb.config.batch_size

        # logger
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

        # tensorbordX log
        logger_tensorboard = SummaryWriter(os.path.join(opt['path']['log']))

        # wandb logger
        wandb.config.update(opt)
        wandb.define_metric("TRAIN/step")
        wandb.define_metric('TRAIN/Learning Rate', step_metric="TRAIN/step")
        wandb.define_metric('TRAIN LOSS/model_DM_loss', step_metric="TRAIN/step")
        wandb.define_metric('TRAIN LOSS/model_DM_loss_image', step_metric="TRAIN/step")
        wandb.define_metric('TRAIN LOSS/model_DM_loss_frequency', step_metric="TRAIN/step")
        wandb.define_metric('TRAIN LOSS/model_DM_loss_preceptual', step_metric="TRAIN/step")
        wandb.define_metric("VAL/step")
        wandb.define_metric('VAL LOSS/model_DM_loss', step_metric="VAL/step")
        wandb.define_metric('VAL LOSS/model_DM_loss_image', step_metric="VAL/step")
        wandb.define_metric('VAL LOSS/model_DM_loss_frequency', step_metric="VAL/step")
        wandb.define_metric('VAL LOSS/model_DM_loss_preceptual', step_metric="VAL/step")
        wandb.define_metric('VAL METRICS/SSIM', step_metric="VAL/step")
        wandb.define_metric('VAL METRICS/PSNR', step_metric="VAL/step")
        wandb.define_metric('VAL METRICS/LPIPS', step_metric="VAL/step")
        wandb.define_metric('VAL METRICS/FID', step_metric="VAL/step")

    # set seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    if 'val' not in list(opt['datasets'].keys()):
        opt['datasets']['val'] = opt['datasets']['test']
    del opt['datasets']['test']

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=False,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=False)

        elif phase == 'val':
            val_set = define_Dataset(dataset_opt)
            val_loader = DataLoader(val_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1,
                                    drop_last=False,
                                    pin_memory=False)

        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)



    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    # define model
    model = define_Model(opt)
    model.init_train()
    # define LPIPS function
    loss_fn_alex = lpips.LPIPS(net='alex').to(model.device)
    # define early stopping
    if opt['train']['is_early_stopping']:
        early_stopping = utils_early_stopping.EarlyStopping(patience=opt['train']['early_stopping_num'])

    # record
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    # if it is not set, keep running
    max_iter_epoch = opt['train']['max_iter_epoch'] if opt['train']['max_iter_epoch'] else 100000000000
    print(f"max iterative step: {max_iter_epoch}")

    for epoch in range(max_iter_epoch):

        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

                # record train loss
                log_wandb = {'TRAIN/step': current_step, 'TRAIN/Learning Rate': model.current_learning_rate(),}
                logger_tensorboard.add_scalar('Learning Rate', model.current_learning_rate(), global_step=current_step)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/model_DM_loss', logs['model_DM_loss'], global_step=current_step)
                log_wandb['TRAIN LOSS/model_DM_loss'] = logs['model_DM_loss']
                if 'model_DM_loss_image' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/model_DM_loss_image', logs['model_DM_loss_image'], global_step=current_step)
                    log_wandb['TRAIN LOSS/model_DM_loss_image'] = logs['model_DM_loss_image']
                if 'model_DM_loss_frequency' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/model_DM_loss_frequency', logs['model_DM_loss_frequency'], global_step=current_step)
                    log_wandb['TRAIN LOSS/model_DM_loss_frequency'] = logs['model_DM_loss_frequency']
                if 'model_DM_loss_preceptual' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/model_DM_loss_preceptual', logs['model_DM_loss_preceptual'], global_step=current_step)
                    log_wandb['TRAIN LOSS/model_DM_loss_preceptual'] = logs['model_DM_loss_preceptual']
                wandb.log(log_wandb)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                # create folder for FID
                # img_dir_tmp_x_t = os.path.join(opt['path']['images'], 'temp_x_t')
                # util.mkdir(img_dir_tmp_x_t)
                # img_dir_tmp_x_start = os.path.join(opt['path']['images'], 'temp_x_start')
                # util.mkdir(img_dir_tmp_x_start)
                # img_dir_tmp_x_direct_recon = os.path.join(opt['path']['images'], 'temp_x_direct_recon')
                # util.mkdir(img_dir_tmp_x_direct_recon)
                # img_dir_tmp_x_recon = os.path.join(opt['path']['images'], 'temp_x_recon')
                # util.mkdir(img_dir_tmp_x_recon)

                # create result dict
                test_results = OrderedDict()
                test_results['direct_recon'] = OrderedDict()
                test_results['direct_recon']['psnr'] = []
                test_results['direct_recon']['ssim'] = []
                test_results['direct_recon']['lpips'] = []
                test_results['recon'] = OrderedDict()
                test_results['recon']['psnr'] = []
                test_results['recon']['ssim'] = []
                test_results['recon']['lpips'] = []

                test_results['model_DM_loss'] = []
                test_results['model_DM_loss_image'] = []
                test_results['model_DM_loss_frequency'] = []
                test_results['model_DM_loss_preceptual'] = []

                for idx, test_data in enumerate(val_loader):
                    pass

    print("Training Stop")


if __name__ == '__main__':

    # pass

    # main()

    ############################## PENDING ##############################

    ############################## TRAINING ##############################

    main("options/CDiffMR/FastMRI/ksu/train_CDiffMR_FastMRIKneePD_m.0.4.s2.ksu.cran.LogSR.d.1.0.cplx.2ch_DEBUG.json")

    ############################## TRAINED ##############################

