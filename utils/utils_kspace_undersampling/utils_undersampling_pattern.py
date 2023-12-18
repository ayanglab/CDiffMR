import os
import time
import random
import numpy as np
from scipy.fftpack import *


def regular_generate_pattern(dim, pe_direction, accel):
    # dim: dimension of mask
    # accel: acceleration rate
    # q
    # seed
    # pf

    if len(dim) == 2:
        if pe_direction == 'row':
            npe = dim[0]
            nro = dim[1]
        elif pe_direction == 'col':
            npe = dim[1]
            nro = dim[0]
        else:
            raise ('Wrong PE Direction!')
        nacq = round(npe / accel)

        mask_1d = []
        for idx in range(npe):
            if not idx % accel:
                mask_1d.append(True)
            else:
                mask_1d.append(False)
        M = np.array(mask_1d)
        M = M[:, np.newaxis]
        M = np.repeat(M, nro, axis=1)
        if pe_direction == 'col':
            M = np.transpose(M, (1, 0))

    elif len(dim) == 3:
        # TODO
        M = 0

    else:
        raise ('Pattern dimension must be 2 or 3.')

    return M


def cs_generate_pattern(dim, pe_direction, accel, q=1, seed=11235813, pf=1):
    # dim: dimension of mask
    # accel: acceleration rate
    # q
    # seed
    # pf

    np.random.seed(seed)

    if len(dim) == 2:
        if pe_direction == 'row':
            npe = dim[0]
            nro = dim[1]
        elif pe_direction == 'col':
            npe = dim[1]
            nro = dim[0]
        else:
            raise ValueError
        nacq = round(npe / accel)
        # nacq = npe // accel
        P = mask_pdf_1d(npe, nacq, q, pf)
        while True:
            temp = np.random.rand(npe)
            M = (temp <= P)
            if sum(M) == nacq:
                break
        M = M[:, np.newaxis]
        M = np.repeat(M, nro, axis=1)
        M = fftshift(M)
        if pe_direction == 'col':
            M = np.transpose(M, (1, 0))
    elif len(dim) == 3:
        # TODO
        M = 0

    else:
        raise ('Pattern dimension must be 2 or 3.')

    return M


def mask_pdf_1d(n, norm, q, pf=1):
    ## VUCSMASKPDF Symmetric array of sampling probabilities
    ks = np.linspace(0, n-1, n) - np.ceil(n/2)
    kmax = np.floor(n/2).astype(int)
    npf = round(pf * n)
    klo = ks[n - npf]
    for kw in np.linspace(1, kmax, kmax):
        P = pdf(ks, kw, klo, q)
        if sum(P) >= norm:
            break
    P = fftshift(P)
    # if n % 2:
    #     a = np.ones(1)
    #     P = np.concatenate((a, P), axis=0)
    return P


def pdf(k, kw, klo, q):

    k = np.where(k < klo, np.inf, k)
    k = np.where(abs(k) < kw, kw, k)
    p = (abs(k)/kw)**(-q)

    return p


def cs_generate_pattern_2d(resolution, accel, sigma=100, seed=0):
    # dim: dimension of mask
    # accel: acceleration rate
    # q
    # seed
    # pf

    np.random.seed(seed)

    if len(resolution) == 2:
        nx = resolution[0]
        ny = resolution[1]

        sampling_rate = 1 / accel
        num_sampling_point = np.ceil(sampling_rate * nx * ny).astype(int)

        assert nx == ny

        P = mask_pdf_2d(resolution, sigma,)
        noise = np.random.rand(nx, ny)

        Pnoise = P + noise

        Pnoise_list = list(Pnoise.reshape(-1))
        Pnoise_list.sort(reverse=True)
        th = Pnoise_list[num_sampling_point -1]
        Pnoise = Pnoise.copy()
        Pnoise[Pnoise >= th] = True
        Pnoise[Pnoise < th] = False
        Pnoise = Pnoise.astype(bool)
        # delta = sum(sum(Pnoise)) - num_sampling_point
        # print(f"found sampling point {sum(sum(Pnoise))}")
        # print(f"required sampling point {num_sampling_point}")
        # print(f"delta sampling point {delta}")


        ###### OLD METHODS #####
        # ths = np.linspace(0, 1, 1000)
        # deltas = []
        # Ps = []
        # for idx, th in enumerate(ths):
        #
        #     P_tmp = Pnoise.copy()
        #     P_tmp[P_tmp >= th] = True
        #     P_tmp[P_tmp < th] = False
        #     delta = sum(sum(P_tmp)) - num_sampling_point
        #     # print(f"---------------------------------------")
        #     # print(f"iterative finding {idx}")
        #     # print(f"found sampling point {sum(sum(P_tmp))}")
        #     # print(f"required sampling point {num_sampling_point}")
        #     # print(f"delta sampling point {delta}")
        #     deltas.append(abs(delta))
        #     Ps.append(P_tmp)
        # idx_fin = np.argmin(deltas)
        # # print(f"final index {idx_fin}")
        # M = Ps[idx_fin]
        # th = ths[idx_fin]

    else:
        raise NotImplementedError

    return Pnoise, th


def mask_pdf_2d(resolution, sigma=100):
    nx = resolution[0]
    ny = resolution[1]
    knx = np.linspace(0, nx - 1, nx) - np.ceil(nx / 2)
    kny = np.linspace(0, ny - 1, ny) - np.ceil(ny / 2)
    kx, ky = np.meshgrid(knx, kny)

    P = pdf2d(kx, ky, sigma)

    return P

def pdf2d(kx, ky, sigma=100):

    exponent = ((abs(kx))**2 + (abs(ky))**2)/(2 * sigma)
    p = np.exp(-exponent)

    return p



if __name__ == '__main__':

    import cv2
    import matplotlib.pyplot as plt
    # raise ValueError('UNCHECKED!')

    # Regular 1D
    # M = regular_generate_pattern((256, 300), 'col', 2)

    # Gaussian 1D
    # M = cs_generate_pattern((256, 300), 'col', 2, q=1, seed=1, pf=1)

    # Project for Diffusion Model
    # Gaussian 2D
    M, _ = cs_generate_pattern_2d((256, 256), accel=2, sigma=100, seed=1)

    # sigma = 100
    seed = 42
    color_map = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
    for i, sigma in enumerate([1, 10, 100]):

        afs = []
        ths = []
        for af in np.linspace(1, 100, 100, endpoint=True):
            if af == 1:
                continue
            # if af != 10:
            #     continue
            print(f'AF: {af}')

            # Gaussain
            M, th = cs_generate_pattern_2d((256, 256), accel=af, sigma=sigma, seed=seed)
            afs.append(af)
            ths.append(th)
            cv2.imwrite(f'/home/jh/MRI_Recon/tmp/mask_temp/mask_af{af}_sigma{sigma}.png', M * 255)

        plt.scatter(afs, ths, c=color_map[i])
    plt.show()

