import numpy as np
import cv2
import matplotlib.pyplot as plt
from matchPics import matchPics
from helper import plotMatches
import scipy


#Q3.5
#Read the image and convert to grayscale, if necessary
x = []
y = []
cv_cover = cv2.imread('../data/cv_cover.jpg')

for i in range(1,36):
	rotation = i*10
	x.append(rotation)
	#Rotate Image
	cv_cover_rotated = scipy.ndimage.rotate(cv_cover, rotation)
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, cv_cover_rotated)
	#Update histogram
	match_count = matches.shape[0]
	y.append(match_count)

	if rotation == 90 or rotation == 180 or rotation == 230:
		plotMatches(cv_cover, cv_cover_rotated, matches, locs1, locs2)


#Display histogram
plt.bar(x, y, width=5)
plt.show()

