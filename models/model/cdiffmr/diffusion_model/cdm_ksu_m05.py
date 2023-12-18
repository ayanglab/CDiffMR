

import torch
from torch import nn
from torch.fft import *
import numpy as np

# Cartesian Mask Support
from utils.utils_kspace_undersampling.fastmri import subsample
# Gaussain 2D Mask Support
from utils.utils_kspace_undersampling import utils_undersampling_pattern, utils_radial_spiral_undersampling


class GaussianDiffusion(nn.Module):
    def __init__(self, opt, denoise_fn, is_train=True,):
        super().__init__()

        self.opt = opt

        self.is_train = is_train
        if is_train:
            self.batch_size = self.opt['datasets']['train']['dataloader_batch_size']
            self.pixel_range = self.opt['datasets']['train']['pixel_range'] if 'pixel_range' in self.opt['datasets']['train'].keys() else '0_1'
        else:
            self.batch_size = self.opt['datasets']['test']['dataloader_batch_size']
            self.pixel_range = self.opt['datasets']['test']['pixel_range'] if 'pixel_range' in self.opt['datasets']['test'].keys() else '0_1'

        # condition
        self.is_dc = self.opt['denoise_fn']['condition']['is_dc']
        self.is_concat = self.opt['denoise_fn']['condition']['is_concat']
        self.is_add = self.opt['denoise_fn']['condition']['is_add']

        # diffusion
        self.diffusion_type = self.opt['diffusion']['diffusion_type']
        self.num_timesteps = self.opt['diffusion']['time_step']
        self.sampling_routine = self.opt['diffusion']['sampling_routine']

        # degradation
        self.ksu_routine = self.opt['diffusion']['degradation']['ksu_routine']
        self.ksu_mask_type = self.opt['diffusion']['degradation']['ksu_mask_type']
        self.ksu_mask_pe = self.opt['diffusion']['degradation']['pe']
        self.ksu_mask_fe = self.opt['diffusion']['degradation']['fe']
        self.ksu_masks = self.get_ksu_masks()

        # denoise
        self.denoise_fn = denoise_fn


    # --------------------------------
    # Degradation
    # --------------------------------
    # return mask generation function
    def get_mask_func(self, ksu_mask_type, af, cf):
        if ksu_mask_type == 'cartesian_regular':
            return subsample.EquispacedMaskFractionFunc(center_fractions=[cf], accelerations=[af])
        elif ksu_mask_type == 'cartesian_random':
            return subsample.RandomMaskFunc(center_fractions=[cf], accelerations=[af])
        elif ksu_mask_type == 'gaussian_2d':
            raise NotImplementedError
            return utils_undersampling_pattern.cs_generate_pattern_2d
        elif ksu_mask_type == 'radial_add':
            return utils_radial_spiral_undersampling.generate_mask_add
        elif ksu_mask_type == 'radial_sub':
            return utils_radial_spiral_undersampling.generate_mask_sub
        elif ksu_mask_type == 'spiral_add':
            return utils_radial_spiral_undersampling.generate_mask_add
        elif ksu_mask_type == 'spiral_sub':
            return utils_radial_spiral_undersampling.generate_mask_sub
        else:
            raise NotImplementedError

    # return the undersampling mask at specific timestep
    def get_ksu_mask(self, ksu_mask_type, af, cf, pe, fe, seed=0):

        mask_func = self.get_mask_func(ksu_mask_type, af, cf)

        if ksu_mask_type in ['cartesian_regular', 'cartesian_random']:

            mask, num_low_freq = mask_func((1, pe, 1), seed=seed)  # mask (torch): (1, pe, 1)
            mask = mask.permute(0, 2, 1).repeat(1, fe, 1)  # mask (torch): (1, pe, 1) --> (1, 1, pe) --> (1, fe, pe)

        elif ksu_mask_type == 'gaussian_2d':
            mask, _ = mask_func(resolution=(fe, pe), accel=af, sigma=100, seed=seed)  # mask (numpy): (fe, pe)
            mask = torch.from_numpy(mask[np.newaxis, :, :])  # mask (torch): (fe, pe) --> (1, fe, pe)

        elif ksu_mask_type in ['radial_add', 'radial_sub', 'spiral_add', 'spiral_sub']:
            sr = 1 / af
            mask = mask_func(mask_type=ksu_mask_type, mask_sr=sr, res=pe, seed=seed)  # mask (numpy): (pe, pe)
            mask = torch.from_numpy(mask[np.newaxis, :, :])  # mask (torch): (pe, pe) --> (1, pe, pe)

        else:
            raise NotImplementedError

        return mask

    # return undersampling masks at different timesteps
    def get_ksu_masks(self):
        masks = []

        if self.ksu_routine == 'LinearSamplingRate':
            # we don't use the sr_list[0]
            sr_list = list(np.linspace(start=0.01, stop=1, num=self.num_timesteps + 1, endpoint=True))[::-1]

            for i in range(self.num_timesteps + 1):
                sr = sr_list[i]
                af = 1 / sr
                cf = sr * 0.32
                masks.append(self.get_ksu_mask(self.ksu_mask_type, af, cf, pe=self.ksu_mask_pe, fe=self.ksu_mask_fe))

        elif self.ksu_routine == 'LogSamplingRate':
            sr_list = list(np.logspace(start=-2, stop=0, num=self.num_timesteps + 1, endpoint=True))[::-1]

            for i in range(self.num_timesteps + 1):
                sr = sr_list[i]
                af = 1 / sr
                cf = sr * 0.32
                masks.append(self.get_ksu_mask(self.ksu_mask_type, af, cf, pe=self.ksu_mask_pe, fe=self.ksu_mask_fe))

        elif self.ksu_mask_type == 'gaussian_2d':
            raise NotImplementedError

        else:
            raise NotImplementedError(f'Unknown k-space undersampling routine {self.ksu_routine}')

        return masks  # remove the first one

    def ksu(self, x_start, mask):

        if self.pixel_range == '0_1':
            pass
        elif self.pixel_range == '-1_1':
            # x_start (-1, 1) --> (0, 1)
            x_start = (x_start + 1) / 2
        elif self.pixel_range == 'complex':
            x_start = torch.complex(x_start[:, :1, ...], x_start[:, 1:, ...])
        else:
            raise ValueError(f"Unknown pixel range {self.pixel_range}.")

        fft = fftshift(fft2(x_start))
        fft = fft * mask
        x_ksu = ifft2(ifftshift(fft))

        if self.pixel_range == '0_1':
            x_ksu = torch.abs(x_ksu)
        elif self.pixel_range == '-1_1':
            x_ksu = torch.abs(x_ksu)
            # x_ksu (0, 1) --> (-1, 1)
            x_ksu = x_ksu * 2 - 1
        elif self.pixel_range == 'complex':
            x_ksu = torch.concat((x_ksu.real, x_ksu.imag), dim=1)
        else:
            raise ValueError(f"Unknown pixel range {self.pixel_range}.")

        return x_ksu


    # --------------------------------
    # Sampling
    # --------------------------------
    @torch.no_grad()
    def sample(self, x_start, x_obs=None, mask=None, batch_size=1, t=None,):

        device = x_start.device
        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        # (C, H, W) --> (B, C, H, W)
        ksu_mask = self.ksu_masks[t].repeat(batch_size, 1, 1, 1).to(device)
        img = self.ksu(x_start=x_start, mask=ksu_mask)

        # x_t
        x_t = img.clone()
        direct_recon = None

        # t: number of timestep
        # t = T, T-1, ..., 1 --> loop (T in total)
        # t = 0 --> out
        while t:
            # print(f'timestep: {t}')
            # step: T-1, T-2, ..., 1, 0 (T in total)
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()

            # condition
            assert (self.is_add and self.is_concat) == 0, 'add and concat can not be used at the same time.'

            img_nocond = img.clone()

            # # C --> C
            # if self.is_add:
            #     img = self.condition_add(x=img, x_obs=x_obs)
            # # C --> 2C
            # if self.is_concat:
            #     img = self.condition_concat(x=img, x_obs=x_obs)

            # img_cond = img.clone()

            # \hat x_0
            img = self.denoise_fn(img, step)

            #  \hat x_0 = R(x_s, s)
            # only in init loop
            if direct_recon == None:
                direct_recon = img.clone()

            # DC
            if self.is_dc:
                assert x_obs is not None, 'missing condition x_obs for data consistency!'
                assert mask is not None, 'missing condition mask for data consistency!'
                img = self.condition_dc(x=img, x_obs=x_obs, mask=mask)

            if self.sampling_routine == 'default':
                # estimate x_{s-1} = D(\hat x_0, s - 1)
                x_times_sub_1 = img.clone()
                ksu_mask_sub_1 = self.ksu_masks[t - 1].repeat(batch_size, 1, 1, 1).to(device)
                x_times_sub_1 = self.ksu(x_start=x_times_sub_1, mask=ksu_mask_sub_1)
                img = x_times_sub_1.clone()

            elif self.sampling_routine == 'x0_step_down':
                # estimate x_{s} = D(\hat x_0, s)
                x_times = img.clone()
                ksu_mask = self.ksu_masks[t].repeat(batch_size, 1, 1, 1).to(device)
                x_times = self.ksu(x_start=x_times, mask=ksu_mask)

                # estimate x_{s-1} = D(\hat x_0, s - 1)
                x_times_sub_1 = img.clone()
                ksu_mask_sub_1 = self.ksu_masks[t - 1].repeat(batch_size, 1, 1, 1).to(device)
                x_times_sub_1 = self.ksu(x_start=x_times_sub_1, mask=ksu_mask_sub_1)

                img = img_nocond - x_times + x_times_sub_1

            else:
                raise NotImplementedError(f'Unknown sampling routine {self.sampling_routine}')

            t = t - 1

        x_recon = img.clone()
        self.denoise_fn.train()

        return x_t, direct_recon, x_recon

    @torch.no_grad()
    def all_sample(self, x_start, x_obs=None, mask=None, batch_size=1, t=None, times=None, eval=True):

        device = x_start.device
        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        if times == None:
            times = t

        # degradation
        ksu_mask = self.ksu_masks[t].repeat(batch_size, 1, 1, 1).to(device)
        img = self.ksu(x_start=x_start, mask=ksu_mask)
        direct_recon = None

        X_0s = []
        X_ts = []
        dict = {}
        dict['ksu_mask'] = []
        dict['X_CHECK_1'] = []
        dict['X_CHECK_2'] = []

        # t: number of timestep
        # t = T, T-1, ..., 1 --> loop (T in total)
        # t = 0 --> out
        while times:
            print(f'timestep: {times}')
            # step: T-1, T-2, ..., 1, 0 (T in total)
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()

            dict['ksu_mask'].append(self.ksu_masks[times].repeat(batch_size, 1, 1, 1).to(device))

            # condition
            assert (self.is_add and self.is_concat) == 0, 'add and concat can not be used at the same time.'

            img_nocond = img.clone()

            X_ts.append(img.detach())

            # # C --> C
            # if self.is_add:
            #     img = self.condition_add(x=img, x_obs=x_obs)
            # # C --> 2C
            # if self.is_concat:
            #     img = self.condition_concat(x=img, x_obs=x_obs)

            # img_cond = img.clone()

            # \hat x_0
            img = self.denoise_fn(img, step)
            X_0s.append(img.detach())

            #  \hat x_0 = R(x_s, s)
            # only in init loop
            if direct_recon == None:
                direct_recon = img.clone()

            # DC
            if self.is_dc:
                assert x_obs is not None, 'missing condition x_obs for data consistency!'
                assert mask is not None, 'missing condition mask for data consistency!'
                img = self.condition_dc(x=img, x_obs=x_obs, mask=mask)

            dict['X_CHECK_1'].append(img)

            if self.sampling_routine == 'default':
                # estimate x_{s-1} = D(\hat x_0, s - 1)
                x_times_sub_1 = img.clone()
                ksu_mask_sub_1 = self.ksu_masks[times - 1].repeat(batch_size, 1, 1, 1).to(device)
                x_times_sub_1 = self.ksu(x_start=x_times_sub_1, mask=ksu_mask_sub_1)
                img = x_times_sub_1.clone()

            elif self.sampling_routine == 'x0_step_down':
                # estimate x_{s} = D(\hat x_0, s)
                x_times = img.clone()
                ksu_mask = self.ksu_masks[times].repeat(batch_size, 1, 1, 1).to(device)
                x_times = self.ksu(x_start=x_times, mask=ksu_mask)

                # estimate x_{s-1} = D(\hat x_0, s - 1)
                x_times_sub_1 = img.clone()
                ksu_mask_sub_1 = self.ksu_masks[times - 1].repeat(batch_size, 1, 1, 1).to(device)
                x_times_sub_1 = self.ksu(x_start=x_times_sub_1, mask=ksu_mask_sub_1)

                img = img_nocond - x_times + x_times_sub_1

            else:
                raise NotImplementedError(f'Unknown sampling routine {self.sampling_routine}')

            dict['X_CHECK_2'].append(img)

            times = times - 1

        X_0s.append(torch.zeros_like(img))
        X_ts.append(img)
        dict['ksu_mask'].append(torch.ones_like(img))
        dict['X_CHECK_1'].append(torch.zeros_like(img))
        dict['X_CHECK_2'].append(torch.zeros_like(img))
        self.denoise_fn.train()

        return X_0s, X_ts, dict


    # --------------------------------
    # degradation for training
    # --------------------------------
    def q_sample(self, x_start, t):

        device = x_start.device

        choose_ksu_mask = [self.ksu_masks[step] for step in t]
        # list (C, H, W) --> (B, C, H, W)
        choose_ksu_mask = torch.stack(choose_ksu_mask).to(device)

        # x_start (B, C, H, W)
        # x_t (B, C, H, W)
        x_t = self.ksu(x_start=x_start, mask=choose_ksu_mask)

        return x_t


    # --------------------------------
    # degradation result for all steps
    # --------------------------------
    @torch.no_grad()
    def q_sample_with_mask(self, x_start, t):

        device = x_start.device

        choose_ksu_mask = [self.ksu_masks[step] for step in t]
        # list (C, H, W) --> (B, C, H, W)
        choose_ksu_mask = torch.stack(choose_ksu_mask).to(device)

        # x_start (B, C, H, W)
        # x_t (B, C, H, W)
        x_t = self.ksu(x_start=x_start, mask=choose_ksu_mask)

        return x_t, choose_ksu_mask

    @torch.no_grad()
    def all_q_sample_with_mask(self, x_start, batch_size=1):

        device = x_start.device
        if eval:
            self.denoise_fn.eval()

        x_ksus = []
        ksu_masks = []
        for step in range(self.num_timesteps):
            t = torch.ones((batch_size), device=device).long() * step
            x_ksu, ksu_mask = self.q_sample_with_mask(x_start=x_start, t=t)
            x_ksus.append(x_ksu)
            ksu_masks.append(ksu_mask)

        return x_ksus, ksu_masks


    # --------------------------------
    # Training forward
    # --------------------------------
    def forward(self, x_start, x_obs=None, mask=None):
        b, c, h, w = x_start.shape
        device = x_start.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Forward
        # x_ksu (B,C/2C,H,W); t: mapping from idx_in_batch to timestep
        x_ksu = self.q_sample(x_start=x_start, t=t)

        # assert (self.is_add and self.is_concat) == 0, 'add and concat can not be used at the same time.'
        # # C --> C
        # if self.is_add:
        #     x_ksu = self.condition_add(x=x_ksu, x_obs=x_obs)
        # # C --> 2C
        # if self.is_concat:
        #     x_ksu = self.condition_concat(x=x_ksu, x_obs=x_obs)

        # Reverse
        # x_recon (B,C/2C,H,W); t: mapping from idx_in_batch to timestep
        x_recon = self.denoise_fn(x_ksu, t)

        # if self.is_dc:
        #     assert x_obs is not None, 'missing condition x_obs for data consistency!'
        #     assert mask is not None, 'missing condition mask for data consistency!'
        #     x_recon = self.condition_dc(x=x_recon, x_obs=x_obs, mask=mask)

        return x_ksu, x_recon

    # ----------------------------------------
    # condition
    # ----------------------------------------
    # concatenate before denoise fn
    def condition_add(self, x, x_obs):
        x = x + x_obs

        return x

    # concatenate before denoise fn
    def condition_concat(self, x, x_obs):
        x = torch.concat((x, x_obs), dim=1)

        return x

    # data consistency after denoise fn
    def condition_dc(self, x, x_obs, mask):

        if self.pixel_range == 'complex':
            x = torch.complex(x[:, :1, ...], x[:, 1:, ...])
            x_obs = torch.complex(x_obs[:, :1, ...], x_obs[:, 1:, ...])

        x_fft = fftshift(fftn(x))
        x_obs_fft = fftshift(fftn(x_obs))
        x_fft_dc = x_obs_fft * mask + x_fft * (1 - mask)
        x = ifftn(ifftshift(x_fft_dc))

        if self.pixel_range == 'complex':
            x = torch.concat((x.real, x.imag), dim=1)
        else:
            x = abs(x)
        return x
