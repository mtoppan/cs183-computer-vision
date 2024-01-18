import numpy as np
import cv2
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from loadVid import loadVid

#Write script for Q4.1

source = loadVid('../data/ar_source.mov')
book = loadVid('../data/book.mov')

cv_cover = cv2.imread('../data/cv_cover.jpg')

result = cv2.VideoWriter('../results/Var.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'), 24, 
                         (book[0].shape[1], book[0].shape[0]))

for i in range(source.shape[0]):
    print(i)

    # trim off the black edges
    source_no_black = source[i][50:source[i].shape[0]-50, :]
   
    # scale the image to be the same height as the book cover
    w_source_scaled = int(np.rint((cv_cover.shape[0] / source_no_black.shape[0]) * source_no_black.shape[1]))
    h_source_scaled = int(np.rint((cv_cover.shape[0] / source_no_black.shape[0]) * source_no_black.shape[0]))

    source_scaled = cv2.resize(source_no_black, (w_source_scaled, h_source_scaled))

    # slice the frame width so it's the same aspect ratio as the book cover, and centered
    w_goal_half = int(np.rint(cv_cover.shape[1] / 2))
    w_source_mid = int(np.rint(source_scaled.shape[1] / 2))

    source_cropped = source_scaled[:h_source_scaled, w_source_mid - w_goal_half:w_source_mid + w_goal_half, :]
    
    matches, locs1, locs2 = matchPics(book[i], cv_cover)
    
    # an attempt to fix the wonky-ness
    if matches.shape[0] < 4: 
        continue

    H2to1, inliers = computeH_ransac(np.take(locs1, matches[:, 0], axis=0), np.take(locs2, matches[:, 1], axis=0))

    composite_img = compositeH(H2to1, book[i], source_cropped)

    result.write(composite_img)

result.release()
