import numpy as np
from scipy.interpolate import RectBivariateSpline

# Worked with Emilie Hopkinson

def LucasKanade(It, It1, rect):

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
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)       
    x1,y1,x2,y2 = rect

    # put your implementation here
    tWidth = It.shape[1]
    tHeight = It.shape[0]   
    heightArray = np.arange(tHeight)
    widthArray = np.arange(tWidth)
    
    rectWidth = int(np.round(x2 - x1))
    rectHeight = int(np.round(y2 - y1))

    splineCalcTemplate = RectBivariateSpline(heightArray, widthArray, It)
    splineCalcImg = RectBivariateSpline(heightArray, widthArray, It1)

    gradient = np.zeros((rectWidth * rectHeight, 2))

    patchX = np.tile(np.linspace(x1, x2, rectWidth, True), (rectHeight, 1))
    patchY = np.tile(np.linspace(y1, y2, rectHeight, True).reshape(-1, 1), (1, rectWidth))

    sTemplate = splineCalcTemplate.ev(patchY, patchX)
    
    for i in range(maxIters):
        newY = patchY + p[1]
        newX = patchX + p[0]
        
        sWarped = splineCalcImg.ev(newY, newX)

        error = sTemplate - sWarped

        gradient[:, 0] = splineCalcImg.ev(newY, newX, dy = 1).ravel()
        gradient[:, 1] = splineCalcImg.ev(newY, newX, dx = 1).ravel()

        J = np.matmul(gradient, np.eye(2))

        H = np.matmul(J.transpose(), J)

        dP = np.linalg.lstsq(H, np.matmul(J.transpose(), error.ravel()), rcond=None)[0]

        p += dP.ravel()

        if threshold > (np.linalg.norm(dP) * np.linalg.norm(dP)):
            break

    return p