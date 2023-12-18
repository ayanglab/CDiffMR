

from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW

from models.select_network import define_denoise_fn
from models.model.cdiffmr.select_diffusion_model import define_diffusion_model
from models.model.model_base import ModelBase
from models.loss import CharbonnierLoss, PerceptualLoss
from models.loss_ssim import SSIMLoss

from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.utils_fourier import *

import wandb
from math import ceil
import copy


class CDiffMR(ModelBase):

    def __init__(self, opt):
        super(CDiffMR, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.opt_dataset = self.opt['datasets']

        # denoise function
        if self.opt['denoise_fn']['condition']['is_concat']:
            opt['denoise_fn']['in_channels'] = opt['denoise_fn']['in_channels'] * 2
        self.denoise_fn = define_denoise_fn(opt)

        # change to select function
        model = define_diffusion_model(opt)
        self.model_DM = model(opt, denoise_fn=self.denoise_fn, is_train=True).to(self.device)
        self.model_DM = self.model_to_device(self.model_DM)

        if self.opt_train['E_decay'] > 0:
            self.model_EMA = model(opt, denoise_fn=self.denoise_fn, is_train=True).to(self.device).eval()

        if opt['rank'] == 0:
            wandb.watch(self.model_DM)


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.model_DM.train()               # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log


    # ----------------------------------------
    # load pre-trained diffusion_model and diffusion_model_ema model
    # ----------------------------------------
    def load(self):
        load_path_model_DM = self.opt['path']['pretrained_model_DM']
        if load_path_model_DM is not None:
            print('Loading model for model_DM [{:s}] ...'.format(load_path_model_DM))
            self.load_network(load_path_model_DM, self.model_DM, strict=self.opt_train['model_DM_param_strict'], param_key='params')

        load_path_model_EMA = self.opt['path']['pretrained_model_EMA']
        if self.opt_train['E_decay'] > 0:
            if load_path_model_EMA is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_model_EMA))
                self.load_network(load_path_model_EMA, self.model_EMA, strict=self.opt_train['model_EMA_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_ema(self.opt_train['E_decay'])
            self.model_EMA.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizer_DM = self.opt['path']['pretrained_optimizer_DM']
        if load_path_optimizer_DM is not None and self.opt_train['model_DM_optimizer_reuse']:
            print('Loading optimizer_DM [{:s}] ...'.format(load_path_optimizer_DM))
            self.load_optimizer(load_path_optimizer_DM, self.model_DM_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.model_DM, 'model_DM', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.model_EMA, 'model_EMA', iter_label)
        if self.opt_train['model_DM_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.model_DM_optimizer, 'optimizer_DM', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        lossfn_type = self.opt_train['lossfn_type']
        if lossfn_type == 'l1':
            self.lossfn = nn.L1Loss().to(self.device)
        elif lossfn_type == 'l2':
            self.lossfn = nn.MSELoss().to(self.device)
        elif lossfn_type == 'l2sum':
            self.lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif lossfn_type == 'ssim':
            self.lossfn = SSIMLoss().to(self.device)
        elif lossfn_type == 'charbonnier':
            self.lossfn = CharbonnierLoss(self.opt_train['charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(lossfn_type))
        self.lossfn_weight = self.opt_train['lossfn_weight']
        self.perceptual_lossfn = PerceptualLoss().to(self.device)


    def total_loss(self):

        self.alpha = self.opt_train['alpha']
        self.beta = self.opt_train['beta']
        self.gamma = self.opt_train['gamma']

        # H HR, E Recon, L LR
        if ('complex' in self.opt_dataset['train']['dataset_type']) and (self.opt_dataset['train']['complex_type'] == '2ch'):
            self.x_start_complex = torch.complex(self.x_start[:, 0:1, :, :], self.x_start[:, 1:, :, :])
            self.x_recon_complex = torch.complex(self.x_recon[:, 0:1, :, :], self.x_recon[:, 1:, :, :])
            self.x_start_1ch = torch.abs(self.x_start_complex)
            self.x_recon_1ch = torch.abs(self.x_recon_complex)
        else:
            self.x_start_complex = self.x_start.clone()
            self.x_recon_complex = self.x_recon.clone()
            self.x_start_1ch = self.x_start.clone()
            self.x_recon_1ch = self.x_recon.clone()

        loss = 0
        if self.alpha:
            self.loss_image = self.lossfn(self.x_recon, self.x_start)
            loss += self.alpha * self.loss_image
        if self.beta:
            self.x_start_k_real, self.x_start_k_imag = fft_map(self.x_start_complex)
            self.x_recon_k_real, self.x_recon_k_imag = fft_map(self.x_recon_complex)
            self.loss_freq = (self.lossfn(self.x_recon_k_real, self.x_start_k_real) + self.lossfn(self.x_recon_k_imag, self.x_start_k_imag)) / 2
            loss += self.beta * self.loss_freq
        if self.gamma:
            self.loss_perc = self.perceptual_lossfn(self.x_recon_1ch, self.x_start_1ch)
            loss += self.gamma * self.loss_perc

        return loss

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        model_DM_optim_params = []
        for k, v in self.model_DM.named_parameters():
            if v.requires_grad:
                model_DM_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        if self.opt_train['model_DM_optimizer_type'] == 'adam':
            self.model_DM_optimizer = Adam(model_DM_optim_params, lr=self.opt_train['model_DM_optimizer_lr'], weight_decay=self.opt_train['model_DM_optimizer_wd'])
        elif self.opt_train['model_DM_optimizer_type'] == 'adamw':
            self.model_DM_optimizer = AdamW(model_DM_optim_params, lr=self.opt_train['model_DM_optimizer_lr'], weight_decay=self.opt_train['model_DM_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['model_DM_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.model_DM_optimizer,
                                                            self.opt_train['model_DM_scheduler_milestones'],
                                                            self.opt_train['model_DM_scheduler_gamma']
                                                            ))
        elif self.opt_train['model_DM_scheduler_type'] == 'MultiStepLRWarmup':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.model_DM_optimizer,
                                                            self.opt_train['model_DM_scheduler_milestones'],
                                                            self.opt_train['model_DM_scheduler_gamma']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):
        self.x_start = data['H'].to(self.device)
        self.x_obs = data['L'].to(self.device)
        self.mask = data['mask'].unsqueeze(1).to(self.device)

    def train_forward(self):

        self.x_ksu, self.x_recon = self.model_DM(x_start=self.x_start,
                                                  x_obs=self.x_obs,
                                                  mask=self.mask)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.current_step = current_step

        # ------------------------------------
        # gradient_accumulation
        # ------------------------------------
        model_DM_gradient_accumulation_every = self.opt_train['model_DM_gradient_accumulation_every'] if self.opt_train['model_DM_gradient_accumulation_every'] else 1

        for i in range(model_DM_gradient_accumulation_every):
            self.train_forward()
            model_DM_loss = self.lossfn_weight * self.total_loss()
            (model_DM_loss / model_DM_gradient_accumulation_every).backward()

        self.model_DM_optimizer.step()
        self.model_DM_optimizer.zero_grad()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        model_DM_optimizer_clipgrad = self.opt_train['model_DM_optimizer_clipgrad'] if self.opt_train['model_DM_optimizer_clipgrad'] else 0
        if model_DM_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['model_DM_optimizer_clipgrad'], norm_type=2)

        # ------------------------------------
        # regularizer
        # ------------------------------------
        model_DM_regularizer_orthstep = self.opt_train['model_DM_regularizer_orthstep'] if self.opt_train['model_DM_regularizer_orthstep'] else 0
        if model_DM_regularizer_orthstep > 0 and current_step % model_DM_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.model_DM.apply(regularizer_orth)
        model_DM_regularizer_clipstep = self.opt_train['model_DM_regularizer_clipstep'] if self.opt_train['model_DM_regularizer_clipstep'] else 0
        if model_DM_regularizer_clipstep > 0 and current_step % model_DM_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.model_DM.apply(regularizer_clip)

        # ------------------------------------
        # record log
        # ------------------------------------
        # the loss recorded here is the final loss_step for gradient accumulation
        self.log_dict['model_DM_loss'] = model_DM_loss.item()
        if self.alpha:
            self.log_dict['model_DM_loss_image'] = self.loss_image.item()
        if self.beta:
            self.log_dict['model_DM_loss_frequency'] = self.loss_freq.item()
        if self.gamma:
            self.log_dict['model_DM_loss_preceptual'] = self.loss_perc.item()

        step_start_ema = self.opt_train['step_start_ema'] if self.opt_train['step_start_ema'] else 0
        update_ema_every = self.opt_train['update_ema_every'] if self.opt_train['update_ema_every'] else 1
        if (self.opt_train['E_decay'] > 0) and (current_step >= step_start_ema) and (current_step % update_ema_every):
            self.update_ema(self.opt_train['E_decay'])


    def record_loss_for_val(self):

        model_DM_loss = self.lossfn_weight * self.total_loss()

        self.log_dict['model_DM_loss'] = model_DM_loss.item()
        if self.alpha:
            self.log_dict['model_DM_loss_image'] = self.loss_image.item()
        if self.beta:
            self.log_dict['model_DM_loss_frequency'] = self.loss_freq.item()
        if self.gamma:
            self.log_dict['model_DM_loss_preceptual'] = self.loss_perc.item()


    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        # self.model_DM.eval()
        with torch.no_grad():
            self.x_t, self.x_direct_recon, self.x_recon = self.model_EMA.sample(x_start=self.x_obs,
                                                                                x_obs=self.x_obs,
                                                                                mask=self.mask,
                                                                                batch_size=1,
                                                                                t=None)
        # self.model_DM.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get x_t, x_start, x_direct_recon, x_recon slice
    # ----------------------------------------
    def current_visuals(self):
        out_dict = OrderedDict()
        out_dict['x_t'] = self.x_t.detach()[0].float().cpu()
        out_dict['x_start'] = self.x_start.detach()[0].float().cpu()
        out_dict['x_direct_recon'] = self.x_direct_recon.detach()[0].float().cpu()
        out_dict['x_recon'] = self.x_recon.detach()[0].float().cpu()
        return out_dict

    def current_visuals_gpu(self):
        out_dict = OrderedDict()
        out_dict['x_t'] = self.x_t.detach()[0].float()
        out_dict['x_start'] = self.x_start.detach()[0].float()
        out_dict['x_direct_recon'] = self.x_direct_recon.detach()[0].float()
        out_dict['x_recon'] = self.x_recon.detach()[0].float()
        return out_dict

    # ----------------------------------------
    # get x_t, x_start, x_direct_recon, x_recon batch
    # ----------------------------------------
    def current_results(self):
        out_dict = OrderedDict()
        out_dict['x_t'] = self.x_t.detach().float().cpu()
        out_dict['x_start'] = self.x_start.detach().float().cpu()
        out_dict['x_direct_recon'] = self.x_direct_recon.detach().float().cpu()
        out_dict['x_recon'] = self.x_recon.detach().float().cpu()
        return out_dict

    def current_results_gpu(self):
        out_dict = OrderedDict()
        out_dict['x_t'] = self.x_t.detach().float()
        out_dict['x_start'] = self.x_start.detach().float()
        out_dict['x_direct_recon'] = self.x_direct_recon.detach().float()
        out_dict['x_recon'] = self.x_recon.detach().float()
        return out_dict



    """
    # ----------------------------------------
    # Information of model_DM
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.model_DM)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.model_DM)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.model_DM)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.model_DM)
        return msg

