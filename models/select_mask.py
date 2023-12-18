
import os
import scipy
import scipy.fftpack
from scipy.io import loadmat
import cv2
import numpy as np


def define_Mask(opt):
    mask_name = opt['mask']

    # 256 * 256 Gaussian 1D
    if mask_name == 'G1D10':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_10.mat"))['maskRS1']
    elif mask_name == 'G1D20':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_20.mat"))['maskRS1']
    elif mask_name == 'G1D30':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_30.mat"))['maskRS1']
    elif mask_name == 'G1D40':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_40.mat"))['maskRS1']
    elif mask_name == 'G1D50':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_50.mat"))['maskRS1']

    # 256 * 256 Gaussian 2D
    elif mask_name == 'G2D10':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_10.mat"))['maskRS2']
    elif mask_name == 'G2D20':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_20.mat"))['maskRS2']
    elif mask_name == 'G2D30':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_30.mat"))['maskRS2']
    elif mask_name == 'G2D40':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_40.mat"))['maskRS2']
    elif mask_name == 'G2D50':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_50.mat"))['maskRS2']

    # 256 * 256 poisson 2D
    elif mask_name == 'P2D10':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_10.mat"))['population_matrix']
    elif mask_name == 'P2D20':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_20.mat"))['population_matrix']
    elif mask_name == 'P2D30':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_30.mat"))['population_matrix']
    elif mask_name == 'P2D40':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_40.mat"))['population_matrix']
    elif mask_name == 'P2D50':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_50.mat"))['population_matrix']

    # 256 * 256 radial
    elif mask_name == 'R10':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_10.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R20':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_20.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R30':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_30.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R40':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_40.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R50':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_50.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R60':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_60.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R70':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_70.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R80':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_80.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R90':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_90.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)

    # 256 * 256 spiral
    elif mask_name == 'S10':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_10.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S20':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_20.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S30':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_30.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S40':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_40.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S50':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_50.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S60':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_60.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S70':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_70.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S80':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_80.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S90':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_90.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)

    # Auto Generated Cartesian Gaussian 1D
    elif mask_name == 'AutoC10':
        mask = 10
    elif mask_name == 'AutoC20':
        mask = 5
    elif mask_name == 'AutoC30':
        mask = 3.3
    elif mask_name == 'AutoC33':
        mask = 3
    elif mask_name == 'AutoC50':
        mask = 2

    # Auto Generated Regular Gaussian 1D (No ACS)
    elif mask_name == 'RegC10':
        mask = 10
    elif mask_name == 'RegC20':
        mask = 5
    elif mask_name == 'RegC30':
        mask = 3.3
    elif mask_name == 'RegC33':
        mask = 3
    elif mask_name == 'RegC50':
        mask = 2

    # GRAPPA-like (with ACS) Regular Acceleration Factor x Central Fraction x PE (from fastMRI)
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe48.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe96.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe128.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe256.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe512.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe48.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe96.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe128.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe256.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe512.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe48.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe96.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe128.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe256.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe512.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe48.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe96.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe128.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe256.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe512.npy'))
        
    # GRAPPA-like (with ACS) Random (Gaussian) Acceleration Factor x Central Fraction x PE (from fastMRI)
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe48.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe96.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe128.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe256.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe512.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe48.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe96.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe128.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe256.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe512.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe48.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe96.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe128.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe256.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe512.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe48.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe96.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe128.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe256.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe512.npy'))

    elif 'radial' in mask_name:
        mask_type, type, sr, res, = mask_name.split('_')
        assert mask_type == 'radial'
        res = int(res[3:])
        mask_pack = np.load(os.path.join('mask', mask_type, f'{mask_type}_res{res}_{type}.npz'))
        mask = mask_pack[f'{mask_type}_{sr}']

    elif 'spiral' in mask_name:
        mask_type, type, sr, res, = mask_name.split('_')
        assert mask_type == 'spiral'
        res = int(res[3:])
        mask_pack = np.load(os.path.join('mask', mask_type, f'{mask_type}_res{res}_{type}.npz'))
        mask = mask_pack[f'{mask_type}_{sr}']

    else:
        raise NotImplementedError('Mask [{:s}] is not defined.'.format(mask_name))

    print('Training model [{:s}] is created.'.format(mask_name))

    return mask
