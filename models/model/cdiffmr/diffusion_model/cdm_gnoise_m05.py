

import torch
from torch import nn
from torch.fft import *
import numpy as np

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
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # denoise
        self.denoise_fn = denoise_fn



    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )


    # --------------------------------
    # Sampling
    # --------------------------------
    @torch.no_grad()
    def sample(self, img, x_obs=None, mask=None, batch_size=1, t=None,):
        # img has been degraded.

        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

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

            x1_bar = self.denoise_fn(img, step)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

            #  \hat x_0 = R(x_s, s)
            # only in init loop
            if direct_recon == None:
                direct_recon = x1_bar.clone()

            xt_bar = x1_bar.clone()
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar.clone()
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img_nocond - xt_bar + xt_sub1_bar
            img = x

            # DC
            if self.is_dc:
                assert x_obs is not None, 'missing condition x_obs for data consistency!'
                assert mask is not None, 'missing condition mask for data consistency!'
                img = self.condition_dc(x=img, x_obs=x_obs, mask=mask)

            t = t - 1

        x_recon = img.clone()
        self.denoise_fn.train()

        return x_t, direct_recon, x_recon

    @torch.no_grad()
    def all_sample(self, img, x_obs=None, mask=None, batch_size=1, t=None, times=None, eval=True):
        # img has been degraded.

        device = img.device
        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        if times == None:
            times = t

        direct_recon = None

        X1_0s = []
        X2_0s = []
        X_ts = []
        dict = {}

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

            # # C --> C
            # if self.is_add:
            #     img = self.condition_add(x=img, x_obs=x_obs)
            # # C --> 2C
            # if self.is_concat:
            #     img = self.condition_concat(x=img, x_obs=x_obs)

            # img_cond = img.clone()

            X_ts.append(img.detach())

            x1_bar = self.denoise_fn(img, step)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

            X1_0s.append(x1_bar.detach())
            X2_0s.append(x2_bar.detach())

            #  \hat x_0 = R(x_s, s)
            # only in init loop
            if direct_recon == None:
                direct_recon = x1_bar.clone()

            xt_bar = x1_bar.clone()
            if times != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar.clone()
            if times - 1 != 0:
                step2 = torch.full((batch_size,), times - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img_nocond - xt_bar + xt_sub1_bar
            img = x

            # DC
            if self.is_dc:
                assert x_obs is not None, 'missing condition x_obs for data consistency!'
                assert mask is not None, 'missing condition mask for data consistency!'
                img = self.condition_dc(x=img, x_obs=x_obs, mask=mask)

            times = times - 1

        self.denoise_fn.train()

        return X1_0s, X2_0s, X_ts, dict


    # --------------------------------
    # Sampling for training
    # --------------------------------
    def q_sample(self, x_start, x_end, t):
        # simply use the alphas to interpolate
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def all_q_sample(self, x_start, x_end, batch_size=1):

        device = x_start.device

        x_mixs = []
        for step in range(self.num_timesteps):
            t = torch.ones((batch_size), device=device).long() * step
            x_mix = self.q_sample(x_start=x_start, x_end=x_end, t=t)
            x_mixs.append(x_mix)

        return x_mixs


    # --------------------------------
    # Training forward
    # --------------------------------
    def forward(self, x_start, x_end, x_obs=None, mask=None):
        b, c, h, w = x_start.shape
        device = x_start.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Forward
        # x_mix (B,C/2C,H,W); t: mapping from idx_in_batch to timestep
        x_mix = self.q_sample(x_start=x_start, x_end=x_end, t=t)

        assert (self.is_add and self.is_concat) == 0, 'add and concat can not be used at the same time.'
        # C --> C
        if self.is_add:
            x_mix = self.condition_add(x=x_mix, x_obs=x_obs)
        # C --> 2C
        if self.is_concat:
            x_mix = self.condition_concat(x=x_mix, x_obs=x_obs)

        # Reverse
        # x_recon (B,C/2C,H,W); t: mapping from idx_in_batch to timestep
        x_recon = self.denoise_fn(x_mix, t)

        if self.is_dc:
            assert x_obs is not None, 'missing condition x_obs for data consistency!'
            assert mask is not None, 'missing condition mask for data consistency!'
            x_recon = self.condition_dc(x=x_recon, x_obs=x_obs, mask=mask)

        return x_mix, x_recon

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

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))