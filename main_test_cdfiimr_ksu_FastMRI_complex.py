'''
# -----------------------------------------
Main Program for Testing
CDiff KSU for MRI_Recon
Dataset: FastMRI
by XXX
# -----------------------------------------
'''

import argparse
import cv2
import csv
import numpy as np
from collections import OrderedDict
import os
import sys
import torch
from utils import utils_image as util
from utils import utils_option as option
from torch.utils.data import DataLoader
import torchvision.utils
from models.model.cdiffmr.diffusion_model.cdm_ksu_m05 import GaussianDiffusion as model
from models.network.cdiffmr.network_cdiff_unet2 import Model as net

from data.select_dataset import define_Dataset
import time
import lpips


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f'create {path}')
    else:
        pass
        # print(f'{path} already exists.')


def main(json_path=''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # set up model
    if os.path.exists(opt['model_path']):
        print(f"loading model from {opt['model_path']}")
    else:
        print('can\'t find model.')

    model = define_model(opt)
    model.eval()
    model = model.to(device)

    # setup folder and path
    save_dir, border = setup(opt)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['lpips'] = []
    test_results['dr_psnr'] = []
    test_results['dr_ssim'] = []
    test_results['dr_lpips'] = []
    test_results['zf_psnr'] = []
    test_results['zf_ssim'] = []
    test_results['zf_lpips'] = []

    with open(os.path.join(save_dir, 'results.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK', 'SSIM', 'PSNR', 'LPIPS'])
    with open(os.path.join(save_dir, 'results_ave.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK',
                         'SSIM', 'SSIM_STD',
                         'PSNR', 'PSNR_STD',
                         'LPIPS', 'LPIPS_STD',
                         'FID'])

    with open(os.path.join(save_dir, 'dr_results.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK', 'SSIM', 'PSNR', 'LPIPS'])
    with open(os.path.join(save_dir, 'dr_results_ave.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK',
                         'SSIM', 'SSIM_STD',
                         'PSNR', 'PSNR_STD',
                         'LPIPS', 'LPIPS_STD',
                         'FID'])

    with open(os.path.join(save_dir, 'zf_results.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK', 'SSIM', 'PSNR', 'LPIPS'])
    with open(os.path.join(save_dir, 'zf_results_ave.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK',
                         'SSIM', 'SSIM_STD',
                         'PSNR', 'PSNR_STD',
                         'LPIPS', 'LPIPS_STD',
                         'FID'])

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    dataset_opt = opt['datasets']['test']
    dataset_opt['pixel_range'] = dataset_opt['pixel_range'] if 'pixel_range' in dataset_opt.keys() else '0_1'

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    for idx, test_data in enumerate(test_loader):

        img_gt = test_data['H'].to(device)
        img_lq = test_data['L'].to(device)
        mask = test_data['mask'].unsqueeze(1).to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()

            time_start = time.time()

            img_t, img_direct_recon, img_recon = model.sample(x_start=img_lq, x_obs=img_lq, mask=mask, batch_size=1, t=opt['diffusion']['samplng_step'])

            time_end = time.time()
            time_c = time_end - time_start  # time used
            print('time cost', time_c, 's')

            # 2 channel --> complex --> abs
            if ('complex' in opt['datasets']['test']['dataset_type']) and (opt['datasets']['test']['complex_type'] == '2ch'):
                img_gt = torch.abs(torch.complex(img_gt[:, :1, ...], img_gt[:, 1:, ...]))
                img_lq = torch.abs(torch.complex(img_lq[:, :1, ...], img_lq[:, 1:, ...]))
                img_direct_recon = torch.abs(torch.complex(img_direct_recon[:, :1, ...], img_direct_recon[:, 1:, ...]))
                img_recon = torch.abs(torch.complex(img_recon[:, :1, ...], img_recon[:, 1:, ...]))
                img_t = torch.abs(torch.complex(img_t[:, :1, ...], img_t[:, 1:, ...]))

            diff_recon_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_recon)), 10)
            diff_direct_recon_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_direct_recon)), 10)
            diff_lq_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_lq)), 10)

        # evaluate lpips recon
        lpips_ = util.calculate_lpips_single(loss_fn_alex, img_gt, img_recon)
        lpips_ = lpips_.data.squeeze().float().cpu().numpy()
        test_results['lpips'].append(lpips_)
        # evaluate lpips direct recon
        dr_lpips_ = util.calculate_lpips_single(loss_fn_alex, img_gt, img_direct_recon)
        dr_lpips_ = dr_lpips_.data.squeeze().float().cpu().numpy()
        test_results['dr_lpips'].append(dr_lpips_)
        # evaluate lpips zf
        zf_lpips_ = util.calculate_lpips_single(loss_fn_alex, img_gt, img_lq)
        zf_lpips_ = zf_lpips_.data.squeeze().float().cpu().numpy()
        test_results['zf_lpips'].append(zf_lpips_)

        # save image
        img_lq = img_lq.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        img_gt = img_gt.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        img_recon = img_recon.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        img_direct_recon = img_direct_recon.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        img_t = img_t.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        diff_recon_x10 = diff_recon_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_direct_recon_x10 = diff_direct_recon_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_lq_x10 = diff_lq_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        # evaluate psnr/ssim recon
        psnr = util.calculate_psnr_single(img_gt, img_recon, border=border)
        ssim = util.calculate_ssim_single(img_gt, img_recon, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

        print('Testing {:d} Recon - PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS: {:.4f} '.format(idx, psnr, ssim, lpips_))

        with open(os.path.join(save_dir, 'results.csv'), 'a') as cf:
            writer = csv.writer(cf)
            writer.writerow(['CDiffMR-KSU', dataset_opt['mask'],
                             test_results['ssim'][idx], test_results['psnr'][idx], test_results['lpips'][idx]])

        # evaluate psnr/ssim direct recon
        dr_psnr = util.calculate_psnr_single(img_gt, img_direct_recon, border=border)
        dr_ssim = util.calculate_ssim_single(img_gt, img_direct_recon, border=border)
        test_results['dr_psnr'].append(dr_psnr)
        test_results['dr_ssim'].append(dr_ssim)
        print('Testing {:d} DirectRecon - PSNR: {:.2f} dB; SSIM: {:.4f};  LPIPS: {:.4f} '.format(idx, dr_psnr, dr_ssim, dr_lpips_))

        with open(os.path.join(save_dir, 'dr_results.csv'), 'a') as cf:
            writer = csv.writer(cf)
            writer.writerow(['CDiffMR-KSU-DR', dataset_opt['mask'],
                             test_results['dr_psnr'][idx], test_results['dr_ssim'][idx], test_results['dr_lpips'][idx]])

        # evaluate psnr/ssim zf
        zf_psnr = util.calculate_psnr_single(img_gt, img_lq, border=border)
        zf_ssim = util.calculate_ssim_single(img_gt, img_lq, border=border)
        test_results['zf_psnr'].append(zf_psnr)
        test_results['zf_ssim'].append(zf_ssim)
        print('Testing {:d} ZF - PSNR: {:.2f} dB; SSIM: {:.4f};  LPIPS: {:.4f} '.format(idx, zf_psnr, zf_ssim, zf_lpips_))

        with open(os.path.join(save_dir, 'zf_results.csv'), 'a') as cf:
            writer = csv.writer(cf)
            writer.writerow(['ZF', dataset_opt['mask'],
                             test_results['zf_ssim'][idx], test_results['zf_psnr'][idx], test_results['zf_lpips'][idx]])

        img_lq = (np.clip(img_lq, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_gt = (np.clip(img_gt, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_recon = (np.clip(img_recon, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_direct_recon = (np.clip(img_direct_recon, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_t = (np.clip(img_t, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8

        diff_recon_x10 = (diff_recon_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_direct_recon_x10 = (diff_direct_recon_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_lq_x10 = (diff_lq_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8

        mkdir(os.path.join(save_dir, 'ZF'))
        mkdir(os.path.join(save_dir, 'GT'))
        mkdir(os.path.join(save_dir, 'Recon'))
        mkdir(os.path.join(save_dir, 'DirectRecon'))
        mkdir(os.path.join(save_dir, 'Different'))

        cv2.imwrite(os.path.join(save_dir, 'ZF', 'ZF_{:05d}.png'.format(idx)), img_lq)
        cv2.imwrite(os.path.join(save_dir, 'GT', 'GT_{:05d}.png'.format(idx)), img_gt)
        cv2.imwrite(os.path.join(save_dir, 'Recon', 'Recon_{:05d}.png'.format(idx)), img_recon)
        cv2.imwrite(os.path.join(save_dir, 'DirectRecon', 'DirectRecon_{:05d}.png'.format(idx)), img_direct_recon)
        cv2.imwrite(os.path.join(save_dir, 'XT', 'XT_{:05d}.png'.format(idx)), img_t)

        diff_recon_x10 = cv2.applyColorMap(diff_recon_x10, cv2.COLORMAP_JET)
        diff_direct_recon_x10 = cv2.applyColorMap(diff_direct_recon_x10, cv2.COLORMAP_JET)
        diff_lq_x10_color = cv2.applyColorMap(diff_lq_x10, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_dir, 'Different', 'Diff_Recon_{:05d}.png'.format(idx)), diff_recon_x10)
        cv2.imwrite(os.path.join(save_dir, 'Different', 'Diff_DirectRecon_{:05d}.png'.format(idx)), diff_direct_recon_x10)
        cv2.imwrite(os.path.join(save_dir, 'Different', 'Diff_ZF_{:05d}.png'.format(idx)), diff_lq_x10_color)

    # summarize psnr/ssim
    ave_psnr = np.mean(test_results['psnr'])
    std_psnr = np.std(test_results['psnr'], ddof=1)
    ave_ssim = np.mean(test_results['ssim'])
    std_ssim = np.std(test_results['ssim'], ddof=1)
    ave_lpips = np.mean(test_results['lpips'])
    std_lpips = np.std(test_results['lpips'], ddof=1)

    print('\n{} \n-- Average PSNR {:.2f} dB ({:.4f} dB)\n-- Average SSIM  {:.4f} ({:.6f})\n-- Average LPIPS  {:.4f} ({:.6f})'
          .format(save_dir, ave_psnr, std_psnr, ave_ssim, std_ssim, ave_lpips, std_lpips))

    # summarize psnr/ssim
    dr_ave_psnr = np.mean(test_results['dr_psnr'])
    dr_std_psnr = np.std(test_results['dr_psnr'], ddof=1)
    dr_ave_ssim = np.mean(test_results['dr_ssim'])
    dr_std_ssim = np.std(test_results['dr_ssim'], ddof=1)
    dr_ave_lpips = np.mean(test_results['dr_lpips'])
    dr_std_lpips = np.std(test_results['dr_lpips'], ddof=1)

    print('\n{} \n-- Average PSNR {:.2f} dB ({:.4f} dB)\n-- Average SSIM  {:.4f} ({:.6f})\n-- Average LPIPS  {:.4f} ({:.6f})'
          .format(save_dir, dr_ave_psnr, dr_std_psnr, dr_ave_ssim, dr_std_ssim, dr_ave_lpips, dr_std_lpips))

    # summarize psnr/ssim zf
    zf_ave_psnr = np.mean(test_results['zf_psnr'])
    zf_std_psnr = np.std(test_results['zf_psnr'], ddof=1)
    zf_ave_ssim = np.mean(test_results['zf_ssim'])
    zf_std_ssim = np.std(test_results['zf_ssim'], ddof=1)
    zf_ave_lpips = np.mean(test_results['zf_lpips'])
    zf_std_lpips = np.std(test_results['zf_lpips'], ddof=1)

    print('\n{} \n-- ZF Average PSNR {:.2f} dB ({:.4f} dB)\n-- ZF Average SSIM  {:.4f} ({:.6f})\n-- ZF Average LPIPS  {:.4f} ({:.6f})'
          .format(save_dir, zf_ave_psnr, zf_std_psnr, zf_ave_ssim, zf_std_ssim, zf_ave_lpips, zf_std_lpips))

    # FID Recon
    log = os.popen("{} -m pytorch_fid {} {} ".format(
        sys.executable,
        os.path.join(save_dir, 'GT'),
        os.path.join(save_dir, 'Recon'))).read()
    print(log)
    fid = eval(log.replace('FID:  ', ''))

    with open(os.path.join(save_dir, 'results_ave.csv'), 'a') as cf:
        writer = csv.writer(cf)
        writer.writerow(['CDiffMR-KSU', dataset_opt['mask'],
                         ave_ssim, std_ssim,
                         ave_psnr, std_psnr,
                         ave_lpips, std_lpips,
                         fid])

    # FID Direct Recon
    log = os.popen("{} -m pytorch_fid {} {} ".format(
        sys.executable,
        os.path.join(save_dir, 'GT'),
        os.path.join(save_dir, 'DirectRecon'))).read()
    print(log)
    dr_fid = eval(log.replace('FID:  ', ''))

    with open(os.path.join(save_dir, 'dr_results_ave.csv'), 'a') as cf:
        writer = csv.writer(cf)
        writer.writerow(['CDiffMR-KSU-DR', dataset_opt['mask'],
                         dr_ave_ssim, dr_std_ssim,
                         dr_ave_psnr, dr_std_psnr,
                         dr_ave_lpips, dr_std_lpips,
                         dr_fid])

    # FID ZF
    log = os.popen("{} -m pytorch_fid {} {} ".format(
        sys.executable,
        os.path.join(save_dir, 'GT'),
        os.path.join(save_dir, 'ZF'))).read()
    print(log)
    zf_fid = eval(log.replace('FID:  ', ''))

    with open(os.path.join(save_dir, 'zf_results_ave.csv'), 'a') as cf:
        writer = csv.writer(cf)
        writer.writerow(['ZF', dataset_opt['mask'],
                         zf_ave_ssim, zf_std_ssim,
                         zf_ave_psnr, zf_std_psnr,
                         zf_ave_lpips, zf_std_lpips,
                         zf_fid])

def define_model(opt):

    if opt['denoise_fn']['condition']['is_concat']:
        opt['denoise_fn']['in_channels'] = opt['denoise_fn']['in_channels'] * 2

    opt_net = opt['denoise_fn']

    denoise_fn = net(opt_net)
    model_DM = model(opt, denoise_fn=denoise_fn, is_train=True)

    param_key = 'params'

    pretrained_model = torch.load(opt['model_path'])
    model_DM.load_state_dict(pretrained_model[param_key] if param_key in pretrained_model.keys() else pretrained_model, strict=True)

    # load the weight from official github
    # pretrained_model_c = torch.load("/home/jh/Cold-Diffusion-Models/recon-diffusion-pytorch/train_results/train_result_cc_train_cartesian_random_LogSamplingRate_100/model.pt")
    # pretrained_model = pretrained_model_c['model']
    # new_pretrained_model = {}
    # for param_key in pretrained_model.keys():
    #     new_param_key = param_key[7:]
    #     new_pretrained_model[new_param_key] = pretrained_model[param_key]
    # model_DM.load_state_dict(new_pretrained_model, strict=True)

    return model_DM


def setup(args):

    save_dir = f"results/{args['task']}/{args['model_name']}"
    border = 0

    return save_dir, border


if __name__ == '__main__':

    # pass
    main()

    ############################## TO DO ##############################

    ############################## WORKSPACE ##############################

    ############################## DONE ##############################

