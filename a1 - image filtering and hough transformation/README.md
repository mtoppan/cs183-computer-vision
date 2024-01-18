# Assignment 1: Image filtering and hough transformation

## Image Filter
This section of the code operates exactly as we discussed convolution filters in class: it flips the given kernel, pads the original image, runs the kernel over the padded image, and replaced values in array with the original’s dimensions with the convolved result of the kernel effects. Of note, the kernel is NOT normalized within this function as doing so might result in error when using such kernels as the Sobel filters that sum to zero. Instead, the kernel must be normalized outside this function. For example, when the Gaussian filter is generated in myEdgeFilter, it is normalized prior to implementation in myImageFilter.

The images below demonstrates the result of the Image Filter when used on img01 with the Gaussian kernel from myEdgeFilter. The first uses a standard deviation (sigma) of 2, while the second uses a sigma of 3 and the third, a sigma of 1. Note the significantly greater blur in the second, demonstrating that a larger filter, in this case, creates greater blur. The fourth and fifth images below demonstrate how heightened and lowered sigmas, respectively, affect the Edge Filter. As one might expect, there are fewer details with the heightened sigma (given that more details are lost in the blur), while the lowered sigma is sharper with significantly more noise.

## Edge Filter


This is where the importance of differences in the ideal parameters for each image became visible, as some visuals at the provided threshold (set in houghScript) were very clean, while others (particularly those scenes with rounded objects and soft, gradient shadows) contained quite a bit of noise. See the difference between the figures below: the first two images compare the given threshold of 0.03 as it affects img01, which is largely blocked, linear areas of color, and img02, which contains more patterns, shadows, and subtleties. The second two demonstrate the effect of an elevated threshold of 0.5 on the same images: the level wipes a significant amount of detail from the second, but a fair amount remains on the first, likely because of the very distinct edges in the image. A threshold of around 0.1 seemed better suited than both of those options to img02 (see the fifth image below).



These results give some indication of the capacity of edge filters such as this: specifically, they show how well-suited such programs are to defining edges in very solidly lit objects with significant color block variation. On the other hand, they also clearly show how difficult it is to work with images that are not well lit, that have significant patterns and/or detail, and that include gradient color and lighting changes rather than better delineated distinctions.

A special shout out to vectorization here! This step was a greater challenge than the others to vectorize, in large part because the non-vectorized option seemed so clear as an answer and was hard to ignore when considering alternatives. However, the use of such imported helpers as cv2.dilate and np.choose served as both an interesting learning opportunity and a gamechanger when it came to runtime. Conducting non-maximum suppression on the matrix as a whole rather than the individual elements drastically cut-down runtime.
## Hough Transform

This script begins by setting a range for the rho and theta values— for the former, based on the maximum distance from the origin, for the latter, based on the range of angles of motion possible. It then initializes an accumulator, populated by zeros and with dimensions based on the length of the ranges for the possible rho and theta values (influenced by the inputs of rhoRes and thetaRes). I then transpose the matrix of nonzero points in the image threshold to get coordinate pairs for non- zero points. Using np.matmul, I multiply the edge matrix (the threshold transpose) and an array of sin and cosine values for all thetas and convert the results to integers s.t. they can be used by np.histogram2d, which tallies up the occurrences of each combination of rho and theta and saves them to the accumulator. np.histogram2d also records the resulting bin sizes, which are equivalent to thetaScale and rhoScale.

The values that are maleable here in the houghScript are rhoRes and thetaRes. Increasing rhoRes (to 4) results in the following Hough visualization and Hough lines:

...while greater thetaRes (to π/30) does this to the Hough visualization and Hough lines, respectively:

## 

This script dilates the given image and parses the resulting image to distinguish cahnges and thereby isolate local maxima. It then gets the rho and theta values for the nLine (or, given number) of local maxima such that lines may be generated. Increasing nLine results in significantly more lines in the image, while lowering it decreases that count. The more lines there are, the less one can distinguish significance as the resulting image is too cluttered and noisy. However, fewer lines sacrifices significant information and risks points of significant light distinction unrelated to the object receiving greater priority. See the first, second, and third images below for 15 nLines, 30, and 5.
