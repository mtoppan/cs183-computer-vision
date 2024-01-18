import numpy as np

def myHoughTransform(img_threshold, rhoRes, thetaRes):
    # compose the range for rho (max distance from the origin)
    w = img_threshold.shape[0]
    h = img_threshold.shape[1]
    d = np.sqrt(w*w + h*h)

    rhos = np.arange(0, 2*d, rhoRes)
    thetas = np.arange(0, 2*(np.pi), thetaRes)

    accumulator = np.zeros((len(rhos), len(thetas)))

    # get the indices for non-zero values in the threshold image
    edges = np.transpose(np.nonzero(img_threshold))

    # for all of the indices in edges and all values in theta, calculate the rho value
    cos = np.cos(thetas)
    sin = np.sin(thetas)

    rho = np.rint(np.matmul(edges, np.array([sin, cos]))).astype(int)

    # remove negative rhos, and remove the corresponding thetas where those negative rhos occur, by limiting the range of the histogram and cropping the resulting image
    accumulator, rBin, tBin = np.histogram2d(rho.ravel(), np.tile(thetas, rho.shape[0]), bins=[rhos,thetas], range=[[0, 1],[0, len(thetas)]])

    res = accumulator[0:accumulator.shape[0] // 2, :]

    return res, rBin, tBin