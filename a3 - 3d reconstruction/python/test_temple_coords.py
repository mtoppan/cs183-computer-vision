import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz
IM1 = io.imread('../data/im1.png')
IM2 = io.imread('../data/im2.png')
SOME_CORRESP = np.load('../data/some_corresp.npz')
P1_CORRESP = SOME_CORRESP['pts1']
P2_CORRESP = SOME_CORRESP['pts2']

# 2. Run eight_point to compute F
F = sub.eight_point(P1_CORRESP, P2_CORRESP, max(IM1.shape[0], IM1.shape[1]))
# hlp.displayEpipolarF(IM1, IM2, F)
print("F:", F)

# 3. Load points in image 1 from data/temple_coords.npz
TEMPLE_COORDS = np.load('../data/temple_coords.npz')
PTS1 = TEMPLE_COORDS['pts1']

# 4. Run epipolar_correspondences to get points in image 2
PTS2 = sub.epipolar_correspondences(IM1, IM2, F, PTS1)
# hlp.epipolarMatchGUI(IM1, IM2, F)

INTRINSICS = np.load('../data/intrinsics.npz')
K1 = INTRINSICS['K1']
K2 = INTRINSICS['K2']
# 5. Compute the camera projection matrix P1
E = sub.essential_matrix(F, K1, K2)

# 6. Use camera2 to get 4 camera projection matrices P2
P2_OPTIONS = hlp.camera2(E)

# # 7. Run triangulate using the projection matrices & figure out the correct P2
I = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = np.zeros((3, 4))
P1 = np.matmul(K1, I)

best_count = 0
best_P3D = None

for z in range(P2_OPTIONS.shape[2]):
    M = P2_OPTIONS[:, :, z]
    pts3D = sub.triangulate(P1, PTS1, np.matmul(K2, M), PTS2)
    if np.count_nonzero(pts3D[:,2] > 0) > best_count:
        best_count = np.count_nonzero(pts3D[:, 2] > 0)
        P2 = np.matmul(K2, M)
        best_P3D = pts3D

print("E:", E)
# print(best_P3D)
# 8. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(best_P3D[:, 0], best_P3D[:, 1], best_P3D[:, 2])
plt.show()

# # 9. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
M2 = np.matmul(np.linalg.inv(K2), P2)
R1 = np.eye(3)
t1 = np.zeros([3,1])
R2 = M2[:,:3]
t2 = M2[:,3:4]
np.savez('../data/extrinsics.npz', R1=R1, t1=t1,R2 = R2, t2 = t2)