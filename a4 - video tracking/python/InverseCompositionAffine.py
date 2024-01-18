import numpy as np
from scipy.interpolate import RectBivariateSpline

# Got confused here and started over rather than using my affine significantly

def InverseCompositionAffine(It, It1, rect):

# 1. Warp image
# 2. Compute error image
# 3. Compute gradient
# 4. Evaluate Jacobian
# 5. Compute Hessian

# 6. Compute delta P
# 7. Update parameters

    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    x1,y1,x2,y2 = rect

    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()

    # # # put your implementation here
    tWidth = It.shape[1]
    tHeight = It.shape[0]   
    heightArray = np.arange(tHeight)
    widthArray = np.arange(tWidth)

    splineCalcTemplate = RectBivariateSpline(heightArray, widthArray, It)
    splineCalcImg = RectBivariateSpline(heightArray, widthArray, It1)

    X, Y = np.meshgrid(widthArray, heightArray)

    gradientX = splineCalcTemplate.ev(Y, X, dy = 1).flatten()
    gradientY = splineCalcTemplate.ev(Y, X, dx = 1).flatten()

    flatX = X.flatten()
    flatY = Y.flatten()
    
    J = np.zeros((len(gradientX), 6))
    
    J[:, 0] = gradientX * flatX
    J[:, 1] = gradientX * flatY
    J[:, 2] = gradientX
    J[:, 3] = gradientY * flatX
    J[:, 4] = gradientY * flatY
    J[:, 5] = gradientY

    for i in range(maxIters):
        tempX = p[0] * X + p[1] * Y + p[2]
        tempY = p[3] * X + p[4] * Y + p[5]
        patchWidth = (x2 > tempX) & (tempX >= x1)
        patchHeight = (y2 > tempY) & (tempY >= y1)
        patch = patchHeight & patchWidth
        tempX = tempX[patch]
        tempY = tempY[patch]

        JPatch = J[patch.flatten()]

        H = np.linalg.inv(np.matmul(JPatch.transpose(), JPatch))

        sWarped = splineCalcImg.ev(tempY, tempX)

        error = sWarped.flatten() - It[patch].flatten()
        warp = np.dot(JPatch.transpose(), error)

        dP = np.dot(H, warp)

        M = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1]])
        dM = np.array([[dP[0], dP[1], dP[2]], [dP[3], dP[4], dP[5]], [0, 0, 1]])

        dM[0, 0] += 1
        dM[1, 1] += 1
        M = np.dot(M, np.linalg.inv(dM))

        p = M[:2, :].flatten()
        
        if threshold > (np.linalg.norm(dP)) * (np.linalg.norm(dP)):
            break

    M = np.array([[p[0], p[1], p[2]],
                 [p[3], p[4], p[5]]]).reshape(2, 3)
    # M = np.array([[1.0+p[0], p[2],    p[4]],
    #              [p[1],     1.0+p[3], p[5]]]).reshape(2, 3)

    return M