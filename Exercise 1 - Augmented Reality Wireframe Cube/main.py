import cv2
import numpy as np
import matplotlib.pyplot as plt

BOARD_COLUMNS = 8
BOARD_ROWS = 5


def poseVectorToTransformationMatrix(pose_vec):
    """

    :param pose_vec: 6 element vector: first 3 elements are axis-angle representation of rotation (w_x, w_y, w_z)
                                   second 3 elements are translation part (t_x, t_y, t_z)
    :return: 3x4 transformation matrix to transform 3D point in world coordinates to point in homogenous
             pixel coordinates (rotation and translation by matrix multiplication)
    """

    # use Rodriques formula to obtain transformation matrix

    rot, trans = np.array(pose_vec[:3]), np.array(pose_vec[3:])

    theta = np.linalg.norm(rot)  # magnitude of rotation
    k = (1/theta) * rot  # unit vector pointing in direction of rotation axis

    # cross-product matrix for the vector k from above
    k_cross = np.array(
        [
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
         ]
    )

    # Rodrigues rotation formula
    rotation_matrix = np.eye(3) + np.sin(theta) * k_cross + (1 - np.cos(theta)) * k_cross @ k_cross

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = trans

    return transform[:3, :]


def projectPoint(points, trans_mat, K):
    """

    :param point: 4D point in homogeneous coordinates from world reference frame
    :param matrix: 3x4 transformation matrix to map from world to camera coordinate frame
    :param point: 3x3 camera intrinsic parameter matrix
    :return: point in pixel coordinates mapped from world frame to image plane
    """

    camera_points = trans_mat @ points
    pixel_points = K @ camera_points[:3, :]

    return pixel_points

# camara intrinsic parameter matrix
K = np.empty(shape=(3, 3), dtype=np.float64)

with open("./data/K.txt", 'r') as intrinsic_f:
    for i, line in enumerate(intrinsic_f):
        K[i] = [float(el) for el in line.split()]

# array that contains the 4D world homogeneous coordinates of the corners of the checkerboard
corners = np.empty(shape=(4, (BOARD_ROWS+1) * (BOARD_COLUMNS+1)), dtype=np.float64)

for i in range(corners.shape[1]):
    corners[:,i] = [(i//(BOARD_ROWS+1))*0.04, (i%(BOARD_ROWS+1))*0.04, 0, 1]
    # print(f'corners[{i}][{j}] = [{corners[i, j]}]')

poses_list = []
with open("./data/poses.txt", 'r') as poses_f:
    for line in poses_f:
        poses_list.append([float(pose) for pose in line.split()])
# array of poses, each line contains the rotation (w_x, w_y, w_z) in axis-angle representation
# and the translation (t_x, t_y, t_z) of the coordinate transformation from world to camera frame as elements
poses = np.array(poses_list)

img = cv2.imread("./data/images_undistorted/img_0001.jpg", cv2.IMREAD_GRAYSCALE)

for pose in poses:
    matrix = poseVectorToTransformationMatrix(pose)
    pixel_coords = projectPoint(corners, matrix, K)
    pixel_coords /= np.repeat(pixel_coords[2:, :], 3, axis=0)

    for i in range(pixel_coords.shape[1]):
        img = cv2.circle(img, (int(pixel_coords[0, i]), int(pixel_coords[1, i])),
                         radius=2, color=(255, 0, 0), thickness=-1)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

