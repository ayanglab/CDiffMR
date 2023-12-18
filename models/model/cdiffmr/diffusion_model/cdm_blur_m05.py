

import torch
from torch import nn
from torch.fft import *
import numpy as np
import torchgeometry as tgm


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
        self.kernel_std = self.opt['diffusion']['degradation']['blur_std']
        self.kernel_size = self.opt['diffusion']['degradation']['blur_size']
        self.blur_routine = self.opt['diffusion']['degradation']['blur_routine']
        self.gaussian_kernels = nn.ModuleList(self.get_kernels())

        # denoise
        self.denoise_fn = denoise_fn


    # --------------------------------
    # Degradation
    # --------------------------------
    # return the 3*3 kernel
    def blur(self, dims, std):
        return tgm.image.get_gaussian_kernel2d(dims, std)

    # return the convolution with Gaussian kernel
    def get_conv(self, dims, std, channels=1, mode='circular'):
        kernel = self.blur(dims, std)  # (3, 3)
        conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=dims, padding=int((dims[0] - 1) / 2),
                         padding_mode=mode,
                         bias=False, groups=channels)
        with torch.no_grad():
            kernel = torch.unsqueeze(kernel, 0)
            kernel = torch.unsqueeze(kernel, 0)
            kernel = kernel.repeat(channels, 1, 1, 1)
            conv.weight = nn.Parameter(kernel)

        return conv

    # return GaussianBlur at different timesteps
    def get_kernels(self):
        kernels = []

        for i in range(self.num_timesteps):
            if self.blur_routine == 'Incremental':  # incremental for std
                kernels.append(self.get_conv((self.kernel_size, self.kernel_size),
                                             (self.kernel_std * (i + 1), self.kernel_std * (i + 1))))
            elif self.blur_routine == 'Constant':  # constant for std
                kernels.append(
                    self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std, self.kernel_std)))
            elif self.blur_routine == 'Constant_reflect':  # constant for std; padding mode: reflect
                kernels.append(
                    self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std, self.kernel_std),
                                  mode='reflect'))
            elif self.blur_routine == 'Exponential_reflect':  # exponential for std; padding model: reflect
                ks = self.kernel_size
                kstd = np.exp(self.kernel_std * i)
                kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode='reflect'))
            elif self.blur_routine == 'Exponential':  # exponential for std;
                ks = self.kernel_size
                kstd = np.exp(self.kernel_std * i)
                kernels.append(self.get_conv((ks, ks), (kstd, kstd)))
            elif self.blur_routine == 'Individual_Incremental':
                ks = 2 * i + 1
                kstd = 2 * ks
                kernels.append(self.get_conv((ks, ks), (kstd, kstd)))
            elif self.blur_routine == 'Special_6_routine':
                ks = 11
                kstd = i / 100 + 0.35
                kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode='reflect'))
            else:
                raise NotImplementedError(f'Unknown blur routine {self.blur_routine}')

        return kernels

    # --------------------------------
    # Sampling
    # --------------------------------
    @torch.no_grad()
    def sample(self, x_start, x_obs=None, mask=None, batch_size=1, t=None,):
        # x_start has not been degraded.

        device = x_start.device
        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        # degradation
        for i in range(t):
            with torch.no_grad():
                img = self.gaussian_kernels[i](x_start)

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
                for i in range(t - 1):
                    with torch.no_grad():
                        x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)
                img = x_times_sub_1.clone()

            elif self.sampling_routine == 'x0_step_down':
                # estimate x_{s} = D(\hat x_0, s)
                x_times = img.clone()
                for i in range(t):
                    with torch.no_grad():
                        x_times = self.gaussian_kernels[i](x_times)

                # estimate x_{s-1} = D(\hat x_0, s - 1)
                x_times_sub_1 = img.clone()
                for i in range(t - 1):
                    with torch.no_grad():
                        x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

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
        for i in range(t):
            with torch.no_grad():
                img = self.gaussian_kernels[i](x_start)

        direct_recon = None

        X_0s = []
        X_ts = []
        dict = {}
        dict['X_CHECK_1'] = []
        dict['X_CHECK_2'] = []

        # t: number of timestep
        # t = T, T-1, ..., 1 --> loop (T in total)
        # t = 0 --> out
        while times:
            print(f'timestep: {times}')
            # step: T-1, T-2, ..., 1, 0 (T in total)
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()

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
                for i in range(times - 1):
                    with torch.no_grad():
                        x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)
                img = x_times_sub_1.clone()

            elif self.sampling_routine == 'x0_step_down':
                # estimate x_{s} = D(\hat x_0, s)
                x_times = img.clone()
                for i in range(times):
                    with torch.no_grad():
                        x_times = self.gaussian_kernels[i](x_times)

                # estimate x_{s-1} = D(\hat x_0, s - 1)
                x_times_sub_1 = img.clone()
                for i in range(times - 1):
                    with torch.no_grad():
                        x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                img = img_nocond - x_times + x_times_sub_1

            else:
                raise NotImplementedError(f'Unknown sampling routine {self.sampling_routine}')

            dict['X_CHECK_2'].append(img)

            times = times - 1

        X_0s.append(torch.zeros_like(img))
        X_ts.append(img)
        dict['X_CHECK_1'].append(torch.zeros_like(img))
        dict['X_CHECK_2'].append(torch.zeros_like(img))
        self.denoise_fn.train()

        return X_0s, X_ts, dict


    # --------------------------------
    # degradation for training
    # --------------------------------
    def q_sample(self, x_start, t):
        # So at present we will for each batch blur it till the max in t.
        # And save it. And then use t to pull what I need. It is nothing but series of convolutions anyway.

        max_iters = torch.max(t)
        all_blurs = []
        x = x_start

        # for each timestep
        for i in range(max_iters + 1):
            with torch.no_grad():
                # blurring
                x = self.gaussian_kernels[i](x)

                # collect the blurred image at timestep i
                # x (B, C, H, W)
                all_blurs.append(x)

        # all_blurs (TimeStep,B,C,H,W)
        # for all the images in batch, xt for all timestep is record
        all_blurs = torch.stack(all_blurs)

        choose_blur = []
        # step is batch size as well so for the 49th step take the step(batch_size)
        # step: idx within a batch
        for step in range(t.shape[0]):
            if step != -1:
                # choose the x_t for idx_in_batch (step) at timestep (t[step])
                choose_blur.append(all_blurs[t[step], step])

            else:
                choose_blur.append(x_start[step])

        # (B, C, H, W)
        choose_blur = torch.stack(choose_blur)

        return choose_blur

    @torch.no_grad()
    def all_q_sample_with_mask(self, x_start, batch_size=1):

        all_blurs = []
        x = x_start

        # for each timestep
        for i in range(self.num_timesteps):
            with torch.no_grad():
                # blurring
                x = self.gaussian_kernels[i](x)

                # collect the blurred image at timestep i
                # x (B, C, H, W)
                all_blurs.append(x)

        return all_blurs


    # --------------------------------
    # Training forward
    # --------------------------------
    def forward(self, x_start, x_obs=None, mask=None):
        b, c, h, w = x_start.shape
        device = x_start.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Forward
        # x_blur (B,C/2C,H,W); t: mapping from idx_in_batch to timestep
        x_blur = self.q_sample(x_start=x_start, t=t)

        assert (self.is_add and self.is_concat) == 0, 'add and concat can not be used at the same time.'
        # C --> C
        if self.is_add:
            x_blur = self.condition_add(x=x_blur, x_obs=x_obs)
        # C --> 2C
        if self.is_concat:
            x_blur = self.condition_concat(x=x_blur, x_obs=x_obs)

        # Reverse
        # x_recon (B,C/2C,H,W); t: mapping from idx_in_batch to timestep
        x_recon = self.denoise_fn(x_blur, t)

        if self.is_dc:
            assert x_obs is not None, 'missing condition x_obs for data consistency!'
            assert mask is not None, 'missing condition mask for data consistency!'
            x_recon = self.condition_dc(x=x_recon, x_obs=x_obs, mask=mask)

        return x_blur, x_recon

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
