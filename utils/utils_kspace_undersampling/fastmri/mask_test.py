import os
import cv2
import numpy as np
import subsample

seed = 0

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'create {path}')
    else:
        print(f'{path} already exists.')


if __name__ == '__main__':

    # ------------------------------------------------------------------
    # project for diffusion model
    # ------------------------------------------------------------------
    #
    # mask_type = 'regular'
    # # cf = 0.08
    # # af = 4
    # pe = 256  # usually 96, 128, 256
    # fe = 256  # should be fixed
    #
    # timesteps = 200
    # sampling_rate_list = list(np.linspace(start=1, stop=0.01, num=timesteps, endpoint=True))
    # af_list = [1 / sampling_rate for sampling_rate in sampling_rate_list]
    # cf_list = [sampling_rate * 0.32 for sampling_rate in sampling_rate_list]
    #
    # print(sampling_rate_list)
    # print(af_list)
    # print(cf_list)
    #
    # for step in range(timesteps):
    #     sampling_rate = sampling_rate_list[step]
    #     af = af_list[step]
    #     cf = cf_list[step]
    #
    #     if mask_type == 'regular':
    #         mask_func = subsample.EquispacedMaskFractionFunc(center_fractions=[cf], accelerations=[af])
    #     elif mask_func == 'random':
    #         mask_func = subsample.RandomMaskFunc(center_fractions=[cf], accelerations=[af])
    #     else:
    #         raise ValueError(f'Unknown mask function: {mask_func}')
    #
    #     mask, num_low_freq = mask_func((1, pe, 1), seed=seed)  # mask (1, pe, 1)
    #     mask = mask[0, :, 0].repeat(fe, 1)  # mask (torch): (1, pe, 1) --> (1, pe) --> (fe, pe)
    #     mask = mask.numpy()
    #
    #     print('Mask Type: {}; SamplingRate: {}; AF: {}; CF: {}; NumLowFreq: {}; PE: {}; FE: {}'.format(mask_type, sampling_rate, af, cf, num_low_freq, pe, fe))
    #
    #     mkdir(os.path.join('./', 'mask_collection', 'cdiffmr'))
    #     # np.save(os.path.join('./', 'mask_collection', 'cdiffmr',f'{mask_type}_af{af}_cf{cf}_pe{pe}.npy'), mask)
    #     cv2.imwrite(os.path.join('./', 'mask_collection', 'cdiffmr', '{}_{:03}_fe256.png'.format(mask_type, step)), mask * 255)
    #     # cv2.imwrite(os.path.join('./', 'mask_collection', 'cdiffmr', '{}_af{}_cf{}_pe{}_fe256.png'.format(mask_type, af, cf, pe)), mask * 255)
    #
    #

    mkdir('./mask_collection/npy/')
    mkdir('./mask_collection/png/')

    afs = [2, 4, 8, 16]
    cfs = [0.16, 0.08, 0.04, 0.02]
    PEs = [512, 320, 256, 128, 96, 48]

    # ------------------------------------------------------------------
    # Random Mask
    # ------------------------------------------------------------------
    for af_idx, af in enumerate(afs):
        cf = cfs[af_idx]
        mask_func = subsample.RandomMaskFunc(center_fractions=[cf], accelerations=[af])
        for pe in PEs:
            mask, _ = mask_func((1, pe, 1), 0, seed)
            mask = mask[0, :, 0].numpy()
            np.save('./mask_collection/npy/random_af{}_cf{}_pe{}.npy'.format(af, cf, pe), mask)
            cv2.imwrite('./mask_collection/png/random_af{}_cf{}_pe{}_fe{}.png'.format(af, cf, pe, pe), np.repeat(mask[np.newaxis, :], pe, axis=0) * 255)

    # ------------------------------------------------------------------
    # Regular Mask
    # ------------------------------------------------------------------
    for af_idx, af in enumerate(afs):
        cf = cfs[af_idx]
        mask_func = subsample.EquispacedMaskFractionFunc(center_fractions=[cf], accelerations=[af])
        for pe in PEs:
            mask, _ = mask_func((1, pe, 1), 0, seed)
            mask = mask[0, :, 0].numpy()
            np.save('./mask_collection/npy/regular_af{}_cf{}_pe{}.npy'.format(af, cf, pe), mask)
            cv2.imwrite('./mask_collection/png/regular_af{}_cf{}_pe{}_fe{}.png'.format(af, cf, pe, pe),
                        np.repeat(mask[np.newaxis, :], pe, axis=0) * 255)
