import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(img_hough, nLines):
    # dilate to isolate local maxima
    dilated = cv2.dilate(img_hough, np.ones((3, 3)))
    imgNMS = np.where(dilated>img_hough, 0, dilated)

    # get the nLine# maxes of the matrix in rhos and thetas
    peaks = np.argpartition(imgNMS.ravel(), -nLines)[-nLines:]
    rhos, thetas = np.unravel_index(peaks, imgNMS.shape)
    # return rhos, thetas
    return rhos, thetas
