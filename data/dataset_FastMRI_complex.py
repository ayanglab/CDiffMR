
import random
import h5py
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_fourier import *
from models.select_mask import define_Mask
from math import floor
from skimage.transform import resize


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f'create {path}')
    else:
        pass
        # print(f'{path} already exists.')


def read_h5(data_path):
    dict = {}
    with h5py.File(data_path, 'r') as file:
        dict['image_complex'] = file['image_complex'][()]
        dict['data_name'] = file['image_complex'].attrs['data_name']
        dict['slice_idx'] = file['image_complex'].attrs['slice_idx']
    return dict


def preprocess_normalisation(img):

    img = img / abs(img).max()

    return img


def undersample_kspace(x, mask, is_noise, noise_level, noise_var):

    # d.1.0.complex --> d.1.1.complex
    # WARNING: This function only take x (H, W), not x (H, W, 1)
    # x (H, W) & x (H, W, 1) return different results
    # x (H, W): after fftshift, the low frequency is at the center.
    # x (H, W, 1): after fftshift, the low frequency is NOT at the center.
    # use abd(fft) to visualise the difference

    fft = fft2(x)
    fft = fftshift(fft)
    fft = fft * mask

    if is_noise:
        raise NotImplementedError
        fft = fft + generate_gaussian_noise(fft, noise_level, noise_var)

    fft = ifftshift(fft)
    x = ifft2(fft)

    return x


def generate_gaussian_noise(x, noise_level, noise_var):
    spower = np.sum(x ** 2) / x.size
    npower = noise_level / (1 - noise_level) * spower
    noise = np.random.normal(0, noise_var ** 0.5, x.shape) * np.sqrt(npower)
    return noise

class DatasetFastMRI(data.Dataset):

    def __init__(self, opt):
        super(DatasetFastMRI, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = self.opt['n_channels']
        self.patch_size = self.opt['H_size']
        self.complex_type = self.opt['complex_type']
        self.is_noise = self.opt['is_noise']
        self.noise_level = self.opt['noise_level']
        self.noise_var = self.opt['noise_var']
        self.is_mini_dataset = self.opt['is_mini_dataset']
        self.mini_dataset_prec = self.opt['mini_dataset_prec']
        self.is_data_in_ram = self.opt['is_data_in_ram']
        self.is_augmentation = self.opt['is_augmentation'] if 'is_augmentation' in self.opt else True

        # get data path of image & sensitivity map
        self.paths_raw = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_raw, 'Error: Raw path is empty.'

        self.paths_H = []
        for path in self.paths_raw:
            if 'file' in path:
                self.paths_H.append(path)
            else:
                raise ValueError('Error: Unknown filename is in raw path')

        self.data_dict = {}
        if self.is_data_in_ram:
            for idx, path_H in enumerate(self.paths_H):
                self.data_dict[f'{idx}'] = read_h5(path_H)
                self.data_dict[f'{idx}']['H_path'] = path_H
        if self.is_mini_dataset:
            pass

        # get mask
        if 'fMRI' in self.opt['mask']:
            mask_1d = define_Mask(self.opt)
            mask_1d = mask_1d[:, np.newaxis]
            mask = np.repeat(mask_1d, 320, axis=1).transpose((1, 0))
            self.mask = mask  # (H, W)
        else:
            self.mask = define_Mask(self.opt)  # (H, W)

        # # save mask
        # import cv2
        # mask_save_path = os.path.join('tmp', '{}.png'.format(self.opt['mask']))
        # cv2.imwrite(mask_save_path, self.mask * 255)
        # print('mask saved in {}'.format(mask_save_path))

    def __getitem__(self, index):

        mask = self.mask  # H, W, 1

        is_noise = self.is_noise
        noise_level = self.noise_level
        noise_var = self.noise_var

        # get gt image
        H_path = self.paths_H[index]
        if self.is_data_in_ram:
            img_dict = self.data_dict[f'{index}']
            assert H_path == img_dict['H_path']
        else:
            img_dict = read_h5(H_path)

        img_H = img_dict['image_complex']  # (H, W) complex

        img_H = preprocess_normalisation(img_H)

        # get zf image
        img_L = undersample_kspace(img_H, mask, is_noise, noise_level, noise_var)

        # expand dim
        img_H = img_H[:, :, np.newaxis]
        img_L = img_L[:, :, np.newaxis]

        # complex to 2 channel
        if self.complex_type == '2ch':
            img_H = np.concatenate((img_H.real, img_H.imag), axis=2)
            img_L = np.concatenate((img_L.real, img_L.imag), axis=2)
        elif self.complex_type == 'abs':
            img_H = np.abs(img_H)
            img_L = np.abs(img_L)
        else:
            raise ValueError(f'Known complex_type {self.complex_type}')

        # get image information
        data_name =img_dict['data_name']
        slice_idx = img_dict['slice_idx']
        img_info = '{}_{:03d}'.format(data_name, slice_idx)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            if self.is_augmentation:
                mode = random.randint(0, 7)
                patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.float2tensor3(patch_L), util.float2tensor3(patch_H)

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.float2tensor3(img_L), util.float2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'mask': mask, 'img_info': img_info}

    def __len__(self):
        return len(self.paths_H)






