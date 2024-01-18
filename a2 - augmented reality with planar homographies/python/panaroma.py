import numpy as np
import cv2
#Import necessary functions
from planarH import computeH_ransac, compositeH
from matchPics import matchPics

#Write script for Q4.2x

l_pano = cv2.imread('../data/dali_left.jpg')
r_pano = cv2.imread('../data/dali_right.jpg')
#find the region of one that corresponds to the other
matches, locs1, locs2 = matchPics(l_pano, r_pano)
#get the homography of those points
H2to1, inliers = computeH_ransac(np.take(locs1, matches[:, 0], axis=0), np.take(locs2, matches[:, 1], axis=0))
#apply the homography to the whole of the second image
w = l_pano.shape[1] + r_pano.shape[1] #should be max width difference between all points
h = l_pano.shape[0] + r_pano.shape[0] #should be max height difference between all points
pano = cv2.warpPerspective(r_pano, H2to1, (w, h))
pano[0:l_pano.shape[0], 0:l_pano.shape[1]] = l_pano

cv2.imshow('image', pano)
#get first point of matching and composite that onto the matrix
cv2.imwrite('../results/pano_img.jpg', pano)
cv2.waitKey(0)
