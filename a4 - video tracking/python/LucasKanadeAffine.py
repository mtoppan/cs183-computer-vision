import numpy as np
from scipy.interpolate import RectBivariateSpline

# Worked with Emilie Hopkinson

def LucasKanadeAffine(It, It1, rect):

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

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    x1,y1,x2,y2 = rect
    # M = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # put your implementation here
    tWidth = It.shape[1]
    tHeight = It.shape[0]   
    heightArray = np.arange(tHeight)
    widthArray = np.arange(tWidth)
    
    rectWidth = int(np.round(x2 - x1))
    rectHeight = int(np.round(y2 - y1))

    splineCalcTemplate = RectBivariateSpline(heightArray, widthArray, It)
    splineCalcImg = RectBivariateSpline(heightArray, widthArray, It1)

    # gradient = np.zeros((rectWidth * rectHeight, 2))

    X, Y = np.meshgrid(np.linspace(y1, y2, rectHeight, True), np.linspace(x1, x2, rectWidth, True))
    
    gradientX = splineCalcTemplate.ev(Y, X, dy = 1).flatten()
    gradientY = splineCalcTemplate.ev(Y, X, dx = 1).flatten()
    gradient = np.vstack((gradientX, gradientY)).transpose()
    # newX = X.flatten()
    # newY = Y.flatten()
    
    # J = np.zeros((len(gradientX), 6))
    
    # J[:, 0] = gradientX * newX
    # J[:, 1] = gradientX * newY
    # J[:, 2] = gradientX
    # J[:, 3] = gradientY * newX
    # J[:, 4] = gradientY * newY
    # J[:, 5] = gradientY

    # sTemplate = splineCalcTemplate.ev(newY, newX)

    for i in range(maxIters):    
        M = np.asarray([[1.0 + p[0], p[2], p[4]], [p[1], 1.0 + p[3], p[5]]]).reshape((2,3)) 
        print(M.shape)
        newX = X.flatten()
        newY = Y.flatten()
    
        J = np.zeros((len(gradientX), 6))
    
        J[:, 0] = gradientX * newX
        J[:, 1] = gradientX * newY
        J[:, 2] = gradientX
        J[:, 3] = gradientY * newX
        J[:, 4] = gradientY * newY
        J[:, 5] = gradientY

        sTemplate = splineCalcTemplate.ev(newY, newX)

        ones = np.ones(newX.shape)

        pointsPrime = np.matmul(M, np.asarray([newY, newX, ones]))
        
        sWarped = splineCalcImg.ev(pointsPrime[1], pointsPrime[0])
        
        error = sTemplate - sWarped
        error = np.reshape(error, (error.shape[0],1,1))

        gradient[:, 0] = splineCalcImg.ev(pointsPrime[1], pointsPrime[0], dy = 1).ravel()
        gradient[:, 1] = splineCalcImg.ev(pointsPrime[1], pointsPrime[0], dx = 1).ravel()

        JTemp = np.zeros((newX.shape[0], 2, 6))

        print(newX.shape)
        JTemp[:, 0, 0] = newX
        JTemp[:, 0, 2] = newY
        JTemp[:, 0, 4] = 1
        JTemp[:, 1, 1] = newX
        JTemp[:, 1, 3] = newY
        JTemp[:, 1, 5] = 1

        J = np.matmul((gradient.reshape((newX.shape[0], 1, 2))), JTemp)

        H = np.sum((np.matmul(J.transpose(0,2,1), J)), axis = 0)

        H = np.linalg.inv(H)

        dP = np.matmul(H, np.sum((np.matmul(J.transpose(0, 2, 1), error)), axis = 0))

        p += dP
        
        if threshold < (np.linalg.norm(dP)) * (np.linalg.norm(dP)):
            break

    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[2],    p[4]],
                 [p[1],     1.0+p[3], p[5]]]).reshape(2, 3)

    return M
