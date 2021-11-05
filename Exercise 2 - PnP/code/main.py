import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimatePoseDLT(p, P, K):
    """

    :param p: matrix of corresponding 2D image points of shape (2, n) where n is the number of 2D-3D correspondences
    :param P: matrix of corresponding 3D world frame point of shape (3, n)
    :param K: camera calibration matrix of shape (3, 3)
    :return: the projections Matrix M = [R | t] for normalized image coordinates
    """

    num_corners = p.shape[1]

    # invert the camera calibration matrix, in order to compute the image-plane normalized pixel coordinates
    K_inv = np.linalg.inv(K)

    # compute image-plance normalized pixel coordinates
    p_norm = K_inv @ np.concatenate((p, np.ones((1, p.shape[1]))), axis=0)
    p_norm = p_norm[0:2, :]

    Q = np.zeros(shape=(2*num_corners, 12), dtype=float)

    for i in range(num_corners):
        Q[2*i, 0] = P[0, i]
        Q[2*i, 1] = P[1, i]
        Q[2*i, 2] = P[2, i]
        Q[2*i, 3] = 1
        Q[2*i, 8] = -p_norm[0, i] * P[0, i]
        Q[2*i, 9] = -p_norm[0, i] * P[1, i]
        Q[2*i, 10] = -p_norm[0, i] * P[2, i]
        Q[2*i, 11] = -p_norm[0, i]

        Q[2*i + 1, 4] = P[0, i]
        Q[2*i + 1, 5] = P[1, i]
        Q[2*i + 1, 6] = P[2, i]
        Q[2*i + 1, 7] = 1
        Q[2*i + 1, 8] = -p_norm[1, i] * P[0, i]
        Q[2*i + 1, 9] = -p_norm[1, i] * P[1, i]
        Q[2*i + 1, 10] = -p_norm[1, i] * P[2, i]
        Q[2*i + 1, 11] = -p_norm[1, i]

    U, S, Vh = np.linalg.svd(Q, full_matrices=True)
    # the solution of the PnP problem is given by the eigenvector corresponding to the smallest eigenvalue of Q'Q
    # which corresponds to the last column (row) of V (V')
    M_tilde_vec = Vh[-1, :]
    M_tilde = np.reshape(M_tilde_vec, (3, 4), order='C')

    # Extract [R|t] with the correct scale from M_tilde ~ [R|t]
    if M_tilde[2, 3] < 0:
        M_tilde *= -1

    R = M_tilde[:, 0:3].copy()

    # Find the closest orthogonal matrix to R
    # % https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    # [U,~,V] = svd(R);
    # R_tilde = U*V';

    U, _, V = np.linalg.svd(R, full_matrices=False)
    R_tilde = U @ V

    # by solving the linear system of the DLT we can only recover the transformation matrix M upto a scale facto
    # M_tilde = [R_tilde | t] = [alpha*R_tilde| alpha*t]

    alpha = np.linalg.norm(R_tilde) / np.linalg.norm(R)
    t = alpha * M_tilde[:, 3]

    M_tilde[:, 0:3] = R_tilde
    M_tilde[:, 3] = t

    return M_tilde


def reprojectPoint(P, M, K):
    """

    :param P: 3D points in world coordinate frame, of shape (3, n)
    :param M: projection matrix M_tilde to map from world to image coordinates, as computed above, shape (3, 4)
    :param K: camera calibration matrix, shape (3, 3)
    :return: reprojected world points onto the image plane
    """

    P = np.concatenate((P, np.ones((1, P.shape[1]))), axis=0)
    reprojected = K @ M @ P  # shape (3, n)
    reprojected /= reprojected[-1:, :]

    return reprojected

K = np.zeros(shape=(3, 3), dtype=float)
with open('../data/K.txt', 'r') as f:
    for i, line in enumerate(f):
        elmts = line.split()
        K[i, 0] = float(elmts[0])
        K[i, 1] = float(elmts[1])
        K[i, 2] = float(elmts[2])

img_points = []
with open('../data/detected_corners.txt', 'r') as f:
    for i, line in enumerate(f):
        elmts = line.split()
        point = np.zeros(shape=(2, 12), dtype=float)
        for j in range(len(elmts)//2):
            point[0, j] = float(elmts[2*j])
            point[1, j] = float(elmts[2*j+1])
        img_points.append(point)

world_points = np.zeros(shape=(3, 12), dtype=float)
with open('../data/p_W_corners.txt', 'r') as f:
    for i, line in enumerate(f):
        elmts = line.split(',')
        world_points[0, i] = float(elmts[0])
        world_points[1, i] = float(elmts[1])
        world_points[2, i] = float(elmts[2])
# convert from cm to m
world_points *= 0.01

for i, point in enumerate(img_points):
    M = estimatePoseDLT(point, world_points, K)
    img = cv2.imread(f'../data/images_undistorted/img_{i+1:04}.jpg', cv2.IMREAD_COLOR)
    reprojected_point = reprojectPoint(world_points, M, K)

    for corner in range(reprojected_point.shape[1]):
        img = cv2.circle(img, (int(reprojected_point[0, corner]), int(reprojected_point[1, corner])),
                         radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()