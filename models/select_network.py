

import functools
import torch
import torchvision.models
from torch.nn import init
import numpy as np


# --------------------------------------------
# Recon Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    if net_type == 'cdiffmr_unet2':
        from models.network.cdiffmr.network_cdiff_unet2 import Model as net
        netG = net(opt_net)


    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG



# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


def define_denoise_fn(opt):
    opt_net = opt['denoise_fn']
    net_type = opt_net['net_type']
    # ----------------------------------------
    # Cold Diff UNet
    # ----------------------------------------
    if net_type == 'cdiffmr_unet':
        raise NotImplementedError
    elif net_type == 'cdiffmr_unet2':
        from models.network.cdiffmr.network_cdiff_unet2 import Model as net
        network = net(opt_net)

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(network,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return network


# --------------------------------------------
# weights initialization
# --------------------------------------------
def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):


    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        # Excecption for ViGU
        EXCP_LIST = ['BasicConv', 'MRConv2d', 'EdgeConv2d', 'GINConv2d', 'GraphSAGE', 'DyGraphConv2d']
        EXCP = np.prod([classname.find(exp) == -1 for exp in EXCP_LIST])

        if (classname.find('Conv') != -1 or classname.find('Linear') != -1) and EXCP:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')

