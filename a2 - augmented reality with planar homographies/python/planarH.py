import numpy as np
import cv2
import random
import math

def computeH(x1, x2):
	rows = x1.shape[0]
	A = np.zeros((rows * 2, 9))
	for i in range(rows):
		x = x2[i][0]
		y = x2[i][1]

		x_prime = x1[i][0]
		y_prime = x1[i][1]

		A[2*i, :] = [-x, -y, -1, 0, 0, 0, x_prime*x, x_prime*y, x_prime]
		A[2*i + 1, :] = [0, 0, 0, -x, -y, -1, y_prime*x, y_prime*y, y_prime]
	u, s, vh = np.linalg.svd(A)
	h = vh[-1]
	H2to1 = h.reshape((3,3))
	return H2to1

def computeH_norm(x1, x2):
	#Q3.7
	#Compute the centroid of the points
	centroid1 = np.mean(x1, axis=0)
	centroid2 = np.mean(x2, axis=0)
	
	#Shift the origin of the points to the centroid
	x1_centered = x1 - centroid1
	x2_centered = x2 - centroid2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	max_distance1 = np.max(np.sqrt(np.sum(x1_centered**2, axis=1)))
	max_distance2 = np.max(np.sqrt(np.sum(x2_centered**2, axis=1)))
	rt2 = np.sqrt(2)

	scale1 = rt2/max_distance1
	scale2 = rt2/max_distance2

	normalized_x1 = x1_centered * scale1
	normalized_x2 = x2_centered * scale2

	#Similarity transform 1
	T1 = np.array([[scale1, 0, -scale1 * centroid1[0]], [0, scale1, -scale1 * centroid1[1]], [0, 0, 1]])

	#Similarity transform 2
	T2 = np.array([[scale2, 0, -scale2 * centroid2[0]], [0, scale2, -scale2 * centroid2[1]], [0, 0, 1]])

	#Compute homography
	normalized_H2to1 = computeH(normalized_x1, normalized_x2)

	#Denormalization
	H2to1 = np.matmul(np.matmul(np.linalg.inv(T1), normalized_H2to1),T2)
	# H2to1 = np.dot(np.linalg.inv(T2), np.dot(normalized_H2to1, T1))
	# print(H2to1)

	return H2to1

def computeH_ransac(x1, x2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	iterations = 1000
	threshold = 10
	test_points = 4
	bestH2to1 = None
	best_inliers= []

	x1 = np.flip(x1, axis=1)
	x2 = np.flip(x2, axis=1)
	# 1. Sample (randomly) the number of points required to fit the model (4?)
	for i in range(iterations):
		# indices = random.sample(range(x1.shape[0]), test_points)
		indices = np.random.choice(len(x1), test_points, replace=False)
		sample_x1 = x1[indices, :]
		sample_x2 = x2[indices, :]
	# 2. Solve for model parameters using samples 
		H2to1 = computeH_norm(sample_x1, sample_x2)
	# 3. Score by the fraction of inliers within a preset threshold of the model
		check_x2 = np.matmul(H2to1, np.vstack((np.transpose(x2), np.ones((x2.shape[0],), dtype=int))))	

		check_x2[2][check_x2[2]<=0] = 1

		check_xs = check_x2[0] / check_x2[2]
		check_ys = check_x2[1] / check_x2[2]
		check = np.transpose(np.vstack((check_xs, check_ys)))

		distances = np.linalg.norm(x1-check, axis=1)
		
		#need points where inliers, and store them in the best_inliers array
		inliers = np.where(distances < threshold, 1, 0)

		if np.sum(inliers) > np.sum(best_inliers):
			best_inliers = inliers
			bestH2to1 = H2to1

	points_to_compute = np.nonzero(best_inliers)
	x1_points = x1[points_to_compute]
	x2_points = x2[points_to_compute]
	bestH2to1 = computeH_norm(x1_points, x2_points)

	return bestH2to1, best_inliers

def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it. But no, we don't! because the order in HarryPotterize is fine!

	#Create mask of same size as template
	mask = np.ones(img.shape[:2], np.uint8)

	#Warp mask by appropriate homography
	mask = cv2.warpPerspective(mask, H2to1, (template.shape[1], template.shape[0]))

	#Warp template by appropriate homography
	warped_img = cv2.warpPerspective(img, H2to1, (template.shape[1], template.shape[0]))
	
	#Use mask to combine the warped template and the image
	composite_img = np.zeros(template.shape, np.uint8)
	composite_img[mask == 1] = warped_img[mask == 1]
	composite_img[mask == 0] = template[mask == 0]
	
	return composite_img


