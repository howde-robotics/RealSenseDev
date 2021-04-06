# Insert your package here
from skimage.color import rgb2xyz
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import lsqr as splsqr
import pdb
from skimage import io

from utils import integrateFrankot, lRGB2XYZ
import numpy as np
from helper import camera2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.optimize
import cv2
from numpy import linalg as LA
from scipy.ndimage.filters import gaussian_filter

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    im1_shape = im1.shape
    sy, sx, _ = im2.shape
    window_size = 15
    buff = int(np.floor(window_size/2))
    window_1 = im1[y1-buff:y1+buff+1, x1-buff:x1+buff+1, :]

    window_gauss = np.zeros((window_size,window_size))
    window_gauss[int(np.floor(window_size/2)),int(np.floor(window_size/2))] = 1
    f = gaussian_filter(window_gauss, 1)
    f = f/np.sum(f)
    f = np.dstack((f,f,f))
    # print(window_1.shape)
    # exit()
    # for y in range(buff,im1.shape[0]-buff):
    #     for x in range(buff,im1.shape[1]-buff):
    xc = x1
    yc = y1
    v = np.array([xc, yc, 1])
    l = F.dot(v)
    s = np.sqrt(l[0]**2+l[1]**2)

    l = l/s

    if l[0] != 0:
        ye = sy-1
        ys = 0
        xe = -(l[1] * ye + l[2])/l[0]
        xs = -(l[1] * ys + l[2])/l[0]
    else:
        xe = sx-1
        xs = 0
        ye = -(l[0] * xe + l[2])/l[1]
        ys = -(l[0] * xs + l[2])/l[1]

    slope = (ye-ys)/(xe-xs)
    b = ye-slope*xe
    smallest_dist = np.inf
    best_x = None
    best_y = None

    for y in range(buff, sy-buff):
        x = int((y-b)/slope)
        window_2 = im2[y-buff:y+buff+1, x-buff:x+buff+1, :]
        dist = np.linalg.norm(np.multiply(f, window_1-window_2))
        if dist < smallest_dist:
            smallest_dist = dist
            best_x = x
            best_y = y

    return best_x, best_y