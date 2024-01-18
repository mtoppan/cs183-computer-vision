import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2):
	#I1, I2 : Images to match
	#Convert Images to GrayScale
	IG1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	IG2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	#Detect Features in Both Images
	features1 = corner_detection(IG1, 0.15)
	features2 = corner_detection(IG2, 0.15)
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(IG1, features1)
	desc2, locs2 = computeBrief(IG2, features2)
	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, 0.65)
	

	return matches, locs1, locs2
