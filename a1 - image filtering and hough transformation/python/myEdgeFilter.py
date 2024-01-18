import numpy as np
from scipy import signal    # For signal.gaussian function
import cv2

from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    # calculate the size of a side of the kernel
    hsize = 2 * np.ceil(3 * sigma) + 1

    # generate the kernel: comes as an array, so reshape to fit the kernel scale
    kernel = signal.gaussian(hsize*hsize, sigma)
    kernel = np.outer(kernel, kernel)

    # normalize gauss kernel if need be
    kernel = kernel / kernel.sum()

    # blur!
    imgc = myImageFilter(img0, kernel)

    # sobel arrays for image gradients
    x_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # convolute to find the horizontal and vertical gradients
    imgx = myImageFilter(imgc, x_sobel)
    imgy = myImageFilter(imgc, y_sobel)

    img1 = np.sqrt(np.square(imgx) + np.square(imgy))

    angles = (np.arctan2(imgy, imgx))

    angles = np.where(angles<0, angles+np.pi, angles) # make sure angle isn't negative
    angles = np.where(angles>np.pi, angles-np.pi, angles) #make sure angle isn't out of the prefd range of 0,180

    angles = np.round(angles/(np.pi/4)).astype(np.int32)

    # denote those values for each angle and ignore the matrix values that don't fit that angle
    angles0 = np.where(((angles == 0) | (angles == 4)), img1, 0)
    angles45 = np.where((angles == 1), img1, 0)
    angles90 = np.where((angles == 2), img1, 0)
    angles135 = np.where((angles == 3), img1, 0)

    # angle kernels
    k0 = np.array([[0, 0, 0],[1, 1, 1],[0, 0, 0]], dtype=np.uint8)
    k45 = np.array([[0, 0, 1],[0, 1, 0],[1, 0, 0]], dtype=np.uint8)
    k90 = np.array([[0, 1, 0],[0, 1, 0],[0, 1, 0]], dtype=np.uint8)
    k135 = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.uint8)

    # dilate the entirety of img1 for each angle kernel
    imgNMS0 = cv2.dilate(img1, k0)
    imgNMS45 = cv2.dilate(img1, k45)
    imgNMS90 = cv2.dilate(img1, k90)
    imgNMS135 = cv2.dilate(img1, k135)

    # for values with angle 0, if the values in k0 are greater than the value in angles0, replace them with 0
    img1_0 = np.where(angles0<imgNMS0, 0, angles0)
    img1_45 = np.where(angles45<imgNMS45, 0, angles45)
    img1_90 = np.where(angles90<imgNMS90, 0, angles90)
    img1_135 = np.where(angles135<imgNMS135, 0, angles135)

    result = np.choose(np.clip(angles,0,3), np.array([img1_0, img1_45, img1_90, img1_135]))

    return result