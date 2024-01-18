import numpy as np
import cv2
#Import necessary functions
from planarH import computeH_ransac, compositeH
from matchPics import matchPics
# from cpselect.cpselect import cpselect

l_pano = cv2.imread('../data/pano_left.jpg')
r_pano = cv2.imread('../data/pano_right.jpg')
#find the region of one that corresponds to the other
# controlpointlist = cpselect("/Users/macytoppan/Downloads/assign2/data/pano_left.jpg", "/Users/macytoppan/Downloads/assign2/data/pano_right.jpg")
# print(controlpointlist)
matches, locs1, locs2 = matchPics(l_pano, r_pano)
#get the homography of those points
H2to1, inliers = computeH_ransac(np.take(locs1, matches[:, 0], axis=0), np.take(locs2, matches[:, 1], axis=0))
#apply the homography to the whole of the second image
width = l_pano.shape[1] + r_pano.shape[1]
height = l_pano.shape[0] + r_pano.shape[0]
result = cv2.warpPerspective(l_pano, H2to1, (width, height))
result[0:r_pano.shape[0], 0:r_pano.shape[1]] = r_pano

cv2.imshow('image', result)
#get first point of matching and composite that onto the matrix
cv2.waitKey(0)

#Write script for Q4.2x
