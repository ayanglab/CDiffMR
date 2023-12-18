import os
import cv2
import scipy
import numpy as np
from math import ceil, floor


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f'create {path}')
    else:
        pass
        # print(f'{path} already exists.')


def load_mask(mask_name):
    # 256 * 256 radial
    if mask_name == 'radial_0':
        mask = np.zeros((256, 256))
    elif mask_name == 'radial_10':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_10.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_20':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_20.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_30':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_30.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_40':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_40.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_50':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_50.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_60':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_60.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_70':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_70.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_80':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_80.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_90':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_90.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'radial_100':
        mask = np.ones((256, 256))

    # 256 * 256 spiral
    elif mask_name == 'spiral_0':
        mask = np.zeros((256, 256))
    elif mask_name == 'spiral_10':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_10.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_20':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_20.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_30':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_30.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_40':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_40.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_50':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_50.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_60':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_60.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_70':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_70.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_80':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_80.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_90':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_90.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'spiral_100':
        mask = np.ones((256, 256))

    else:
        raise ValueError(f'Invalid mask name: {mask_name}')

    return mask


def generate_mask_sub(mask_type, mask_sr, res, seed=0):

    mask_type, type = mask_type.split('_')

    # percentage
    mask_sr = mask_sr * 100

    # set random seed
    np.random.seed(seed)

    mask_name = f'{mask_type}_{mask_sr}'

    # load template
    mask_sr_temp = int(ceil(mask_sr / 10) * 10)
    mask_name_temp = f'{mask_type}_{mask_sr_temp}'
    mask_temp = load_mask(mask_name_temp)

    # adjust resolution
    mask_temp = cv2.resize(mask_temp, (res, res), interpolation=cv2.INTER_NEAREST)  # resize
    mask_temp = (mask_temp > 0.5).astype(np.float32)  # threshold to 0-1 mask

    # remove sampling point to match the undersampling rate
    sampling_pixel_num_theo = int(res * res * mask_sr / 100)
    sampling_pixel_num_init = int(sum(sum(mask_temp)))

    mask_fin = mask_temp.copy()
    if sampling_pixel_num_init > sampling_pixel_num_theo:
        # print(f'Processing {mask_name}: {sampling_pixel_num_init} --> {sampling_pixel_num_theo}. Removing sampling points...')

        nonzero_index = np.where(mask_temp > 0)  # get mask_temp non-zero pixel index
        # randomly remove non-zero pixels to match sampling_pixel_num_theo
        nonzero_index_remove_ids = np.random.choice(np.arange(sampling_pixel_num_init),
                                                    sampling_pixel_num_init - sampling_pixel_num_theo, replace=False)

        for nonzero_index_remove_id in list(nonzero_index_remove_ids):
            mask_fin[nonzero_index[0][nonzero_index_remove_id], nonzero_index[1][nonzero_index_remove_id]] = 0
    else:
        # print(f'Processing {mask_name}: {sampling_pixel_num_init} -x-> {sampling_pixel_num_theo}.')
        pass

    return mask_fin



def generate_mask_add(mask_type, mask_sr, res, seed=0):

    mask_type, type = mask_type.split('_')

    # percentage
    mask_sr = mask_sr * 100

    # set random seed
    np.random.seed(seed)

    mask_name = f'{mask_type}_{mask_sr}'

    # load template
    mask_sr_temp = int(floor(mask_sr / 10) * 10)
    mask_name_temp = f'{mask_type}_{mask_sr_temp}'
    mask_temp = load_mask(mask_name_temp)

    # adjust resolution
    mask_temp = cv2.resize(mask_temp, (res, res), interpolation=cv2.INTER_NEAREST)  # resize
    mask_temp = (mask_temp > 0.5).astype(np.float32)  # threshold to 0-1 mask

    # remove sampling point to match the undersampling rate
    sampling_nonzero_pixel_num_theo = res * res - int(res * res * mask_sr / 100)
    sampling_nonzero_pixel_num_init = res * res - int(sum(sum(mask_temp)))

    mask_fin = mask_temp.copy()
    if sampling_nonzero_pixel_num_init > sampling_nonzero_pixel_num_theo:
        # print(f'Processing {mask_name}: {sampling_pixel_num_init} --> {sampling_pixel_num_theo}. Removing sampling points...')

        nonzero_index = np.where(mask_temp == 0)  # get mask_temp non-zero pixel index
        # randomly add non-zero pixels to match sampling_pixel_num_theo
        nonzero_index_remove_ids = np.random.choice(np.arange(sampling_nonzero_pixel_num_init),
                                                    sampling_nonzero_pixel_num_init - sampling_nonzero_pixel_num_theo, replace=False)

        for nonzero_index_remove_id in list(nonzero_index_remove_ids):
            mask_fin[nonzero_index[0][nonzero_index_remove_id], nonzero_index[1][nonzero_index_remove_id]] = 1
    else:
        # print(f'Processing {mask_name}: {sampling_pixel_num_init} -x-> {sampling_pixel_num_theo}.')
        pass

    return mask_fin


if __name__ == '__main__':

    # # set random seed
    # np.random.seed(0)

    # loop for type
    for type in ['sub', 'add']:

        # loop for mask_type
        for mask_type in ['radial', 'spiral']:

            # loop for resolution
            for resolution in [256, 320, 512]:
                mask_sr_list = np.linspace(1, 100, 100, dtype=int)  # SR: 0.01-1 (x100)
                mask_dict = {}

                # loop for sr
                for mask_sr in mask_sr_list:

                    # mask name
                    mask_name = f'{mask_type}_{mask_sr}'

                    # generate mask
                    if type == 'sub':
                        mask_fin = generate_mask_sub(f'{mask_type}_{type}', mask_sr * 0.01, resolution)
                    elif type == 'add':
                        mask_fin = generate_mask_add(f'{mask_type}_{type}', mask_sr * 0.01, resolution)
                    else:
                        raise ValueError(f'Invalid mask type: {type}')

                    # save mask to dict
                    mask_dict[mask_name] = mask_fin

                    # save mask to file
                    mkdir(os.path.join('tmp', f'{mask_type}_{type}'))
                    cv2.imwrite(os.path.join('tmp', f'{mask_type}_{type}', f'{mask_name}.png'), mask_fin * 255)

                # save mask dict using npz
                np.savez(os.path.join('mask', mask_type, f'{mask_type}_res{resolution}_{type}.npz'), **mask_dict)
