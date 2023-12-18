import torch
from torch import nn
import os
import cv2
import gc
import numpy as np
from scipy.io import *
from scipy.fftpack import *



# Fourier Transform
def fft_map(x):
    fft_x = torch.fft.fftn(x)
    fft_x_real = fft_x.real
    fft_x_imag = fft_x.imag

    return fft_x_real, fft_x_imag

