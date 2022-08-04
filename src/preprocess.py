import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def preprocess(imgs):
    imgs = imgs[:, :195]
    imgs = rgb2gray(imgs)
    imgs = imgs[:, ::2, ::2]
    imgs = np.expand_dims(imgs, axis=1)
    return imgs


@jit(nopython=True)
def rgb2gray(imgs):
    return 0.2989 * imgs[:, :, :, 2] + 0.5870 * imgs[:, :, :, 1] + 0.1140 * imgs[:, :, :, 0]
