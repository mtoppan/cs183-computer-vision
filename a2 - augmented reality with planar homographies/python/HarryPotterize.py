import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q3.9
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_desk, cv_cover)

H2to1, inliers = computeH_ransac(np.take(locs1, matches[:, 0], axis=0), np.take(locs2, matches[:, 1], axis=0))

rescaled_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
warped = cv2.warpPerspective(hp_cover, H2to1, (cv_desk.shape[1], cv_desk.shape[0]))

composite_img = compositeH(H2to1, cv_desk, rescaled_cover)

cv2.imshow('image', composite_img)

cv2.imwrite('../results/composite_img.jpg', composite_img)
cv2.waitKey(0)