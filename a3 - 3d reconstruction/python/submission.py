"""
Homework 5
Submission Functions
"""
# import packages here
import numpy as np
import helper as hlp
import scipy.signal

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    #Divide each coordinate by M using transformation matrix T
    T = np.array([[1/M, 0], [0, 1/M]])
    PTS1 = np.matmul(pts1, T)
    PTS2 = np.matmul(pts2, T)
    # 1. Construct the M x 9 matrix A
    #xx' xy' x yx' yy' y x' y' 1
    rows = PTS1.shape[0]
    A = np.zeros((rows, 9))
    for i in range(rows):
        x = PTS2[i][0]
        y = PTS2[i][1]
        x_prime = PTS1[i][0]
        y_prime = PTS1[i][1]
        A[i, :] = [x*x_prime, x*y_prime, x, y*x_prime, y*y_prime, y, x_prime, y_prime, 1]
    # 2. Find the SVD of A
    u, sigma, v = np.linalg.svd(A)
    # 3. Entries of F are the elements of column of V corresponding to the least singular value
    F = v[-1].reshape((3,3))
    #Decompose F with SVD to get the three matrices, then set the smallest singular value in sigma to zero for sigma_prime, then F' = u sigma_prime vtranspose
    u_new, sigma_new, v_new = np.linalg.svd(F)
    sigma_min = np.zeros((u_new.shape[1], v_new.shape[0]))
    sigma_min[:2, :2] = np.diag(sigma_new)[:2, :2]
    F_PRIME = np.matmul(np.matmul(u_new, sigma_min), v_new)
    #call refineF before unscaling
    F_REFINED = hlp.refineF(F_PRIME, PTS1, PTS2)
    #Unscale F (TtransposeFT)
    T_NEW = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F_FINAL = np.matmul(np.matmul(T_NEW.transpose(), F_REFINED), T_NEW)
    return F_FINAL


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # replace pass by your implementation
    #to match one point x in img1, use F to estimate the corresponding epipolar line l' and generate a set of candidate points in the second image
    w = 20
    kernel = 30

    pts1 = (np.hstack((pts1, np.ones((pts1.shape[0], 1)))))
    lines = np.matmul(F, (pts1.transpose())).transpose()
    # print(lines)

    im1 = np.pad(im1, ((w, w), (w, w), (0,0)))
    im2 = np.pad(im2, ((w, w), (w, w), (0,0)))

    pts2 = []

    for i, (x, y, z) in enumerate(pts1):

        y = int(y)
        x = int(x)

        m = -(lines[i][0] / lines[i][1])
        b = -(lines[i][2] / lines[i][1])

        best_point = [0,0]
        best_dist = 100000000000

        for x_cand in range(x - (w//2), x + (w//2)):
            y_cand = int((m * x_cand) + b)

            kern_x1_min = x - (kernel//2)
            kern_x2_min = x_cand - (kernel//2)

            kern_y1_min = y - (kernel//2)
            kern_y2_min = y_cand - (kernel//2)

            kern_x1_max = x + (kernel//2)
            kern_x2_max = x_cand + (kernel//2)

            kern_y1_max = y + (kernel//2)
            kern_y2_max = y_cand + (kernel//2)

            kern1 = im1[kern_y1_min:kern_y1_max, kern_x1_min:kern_x1_max]
            kern2 = im1[kern_y2_min:kern_y2_max, kern_x2_min:kern_x2_max]

            dist = np.linalg.norm(kern1-kern2)

            if dist < best_dist:
                best_dist = dist
                best_point = [x_cand, y_cand]

        pts2.append(best_point)
    pts2 = np.asarray(pts2)
    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation 
    E = np.matmul(np.matmul(K2.transpose(), F), K1)
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    P = []

    for i in range(pts1.shape[0]):
        r1 = pts1[i][1]*(P1[2].transpose()) - (P1[1].transpose())
        r2 = (P1[0].transpose()) - pts1[i][0]*(P1[2].transpose())
        r3 = pts2[i][1]*(P2[2].transpose()) - (P2[1].transpose())
        r4 = (P2[0].transpose()) - pts2[i][0]*(P2[2].transpose())

        A = np.array([r1, r2, r3, r4])
        u, sigma, vt = np.linalg.svd(A)
        # v = vt.transpose()
        X = vt[-1].reshape(4,1)
        X = X/X[-1]
        P.append(X)
    pts3d = np.asarray(P)
    # print(pts3d.shape)

    #reprojection error
    test_pts1 = np.matmul(P1, (pts3d.reshape(288, 4).transpose()))
    test_pts1 = ((test_pts1/test_pts1[-1]).transpose())[:, :-1]
    r_err = np.linalg.norm(pts1-test_pts1) / pts3d.shape[0]
    print("r_err:", r_err)

    pts3d = pts3d[:, :-1]

    return pts3d


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    c1 = -(np.matmul((np.linalg.inv(np.matmul(K1, R1))), np.matmul(K1, t1))) 
    c2 = -(np.matmul((np.linalg.inv(np.matmul(K2, R2))), np.matmul(K2, t2))) 

    r1 = ((c1-c2)/np.linalg.norm(c1-c2)).transpose()
    r2 = np.cross(R1[2].reshape(3,1).transpose(), r1)
    r3 = np.cross(r2, r1)

    R_NEW = np.array([r1.reshape(3,), r2.reshape(3,), r3.reshape(3,)])

    R1p = R_NEW
    R2p = R_NEW

    K1p = K2
    K2p = K2

    t1p = -(np.matmul(R_NEW, c1))
    t2p = -(np.matmul(R_NEW, c2))

    M1 = np.matmul(np.matmul(K1p, R1p), np.linalg.inv(np.matmul(K1, R1)))
    M2 = np.matmul(np.matmul(K2p, R2p), np.linalg.inv(np.matmul(K2, R2)))

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    w = (win_size-1)//2
    im1p = np.pad(im1, ((0,0), (max_disp, max_disp)))
    c = np.ones((win_size, win_size))

    dispM = np.zeros((im1.shape[0], im2.shape[1]))
    for x in range (im1.shape[1]):
        # this is from an earlier iteration when i used convolve2d. however, i had a hard time getting the desired results with it, so abandoned in for a longer form method. :(
        # x = (im2 - im1p[:, max_disp-z:max_disp-z+im1.shape[1]])**2
        # sums[:, :, z] = scipy.signal.convolve2d(x, c)[w:w+im1.shape[0], w:w+im1.shape[1]]
        for y in range(im1.shape[0]):
            dist = []
            for z in range(max_disp+1):
                total = 0
                for i in range (-w, w+1):
                    for j in range(-w, w+1):
                        try:
                            total += (im1[y+i-1][x+j-1] - im2[y+i-1][x+j-(z+1)])**2
                        except:
                            pass
                dist.append(total)
            dispM[y][x] = np.argmin(dist)

    
    # dispM = np.argmin(sums, axis=2)
    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    f = K1[0][0]
    c1 = -np.linalg.inv((np.matmul(K1,R1)))@np.matmul(K1,t1)
    c2 = -np.linalg.inv((np.matmul(K2,R2)))@np.matmul(K2,t2)
    b = np.linalg.norm(c1-c2)

    depthM = np.zeros_like(dispM)
    depthM[dispM!=0] = (b*f) / dispM[dispM!=0]

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    A = []
    for i in range(len(x)):
        A.append([X[i,0], X[i,1], X[i,2], 1, 0, 0, 0, 0, -x[i][0]*X[i,0], -x[i][0]*X[i,1], -x[i][0]*X[i,2], -x[i][0]])
        A.append([0, 0, 0, 0, X[i,0], X[i,1], X[i,2], 1, -x[i][1]*X[i,0], -x[i][1]*X[i,1], -x[i][1]*X[i,2], -x[i][1]])
    u, s, vh = np.linalg.svd(A)
    P = vh[-1].reshape((3,4))
    return P


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    u, s, vh = np.linalg.svd(P)
    c = vh[-1]
    c = c/c[-1]
    c = c[:3]

    M = P[:, :3]
    K,R = scipy.linalg.rq(M)

    t = -(np.matmul(R, c))

    return K, R, t
