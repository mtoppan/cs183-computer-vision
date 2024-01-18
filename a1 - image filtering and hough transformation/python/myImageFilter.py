import numpy as np

def myImageFilter(img0, h):
    # flip it!
    kernel = np.flip(h)
        
    # for padding, get half -1 of kernel size
    w = int((kernel.shape[0]-1)/2)

    # copy to fill in with filtered values
    img1 = np.zeros_like(img0)

    # pad it up! edge-based, rather than with all 0s
    img_padded = np.pad(img0, ((w,w),(w,w)), mode='edge')

    # loop through, applying the kernel to each spot
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            img1[x, y] = np.sum(kernel * img_padded[x: (x + kernel.shape[0]), y: (y + kernel.shape[1])])

    return img1
    

#     Write a function that convolves an image with a given convolution filter
# img1 = myImageFilter(img0, h)
# As input, the function takes a greyscale image (img0) and a convolution filter stored in matrix h. The output of the function should be an image img1 of the same size as img0 which results from convolving img0 with h. You can assume that the filter h is odd sized along both dimensions. You will need to handle boundary cases on the edges of the image. For example, when you place a convolution mask on the top left corner of the image, most of the filter mask will lie outside the image. One solution is to output a zero value at all these locations, the better thing to do is to pad the image such that pixels lying outside the image boundary have the same intensity value as the nearest pixel that lies inside the image.
# You can call NumPy’s pad function to pad the array; read about the different modes available for padding HERE. You can also use the other numpy functions such as the ROLL function. However, your code cannot call on convolve, correlate, fftconvolve, or any other similar functions. You may compare your output to these functions for comparison and debugging.
# This function should be vectorized, meaning that you should be relying as much as possible on mathematical functions that operate on vectors and matrices. Avoid in- dexing into matrix elements one at a time (or do so sparingly), as this will significantly slow down your code. Examples and meaning of vectorization can be found HERE.
# Specifically, try to reduce the number of for loops that you use in the function as much as possible (two for loops is sufficient to implement a fast version of convolution).

# test_image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 90, 90, 90, 90, 90, 0, 0],
#                        [0, 0, 0, 90, 90, 90, 90, 90, 0, 0],
#                        [0, 0, 0, 90, 0, 90, 90, 90, 0, 0],
#                        [0, 0, 0, 90, 90, 90, 90, 90, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 90, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# test_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# print(myImageFilter(test_image, test_filter))