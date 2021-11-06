import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt

BOARD_COLUMNS = 8
BOARD_ROWS = 5

def distortPoints(point, D, K):
    """"
    :param point: tuple containing the pixel coordinates in x and y direction (x,y)
    :param D: matrix containing the parameters of the distortion model D = [k1, k2]
    :return: tuple with distored pixel coordinates in x and y direction
    """

    # account for lens distortion
    principal_point = np.array(K[:2, 2])
    if len(point.shape) > 1:
        principal_point = np.repeat(np.expand_dims(principal_point, 1), point.shape[1], axis=1)
    # distance of point in image from principal point
    point_diff = point - principal_point
    r = np.linalg.norm(point_diff, axis=0, keepdims=True)
    distorted_points = (1 + D[0, 0] * (r ** 2) + D[0, 1] * (r ** 4)) * point_diff + principal_point

    return distorted_points


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

    # normalize homogeneous coordinates
    pixel_points = pixel_points[:2, :] / pixel_points[2:, :]

    pixel_points = distortPoints(pixel_points, D, K)

    return pixel_points


# camera intrinsic parameter matrix
K = np.empty(shape=(3, 3), dtype=np.float64)

with open("./data/K.txt", 'r') as intrinsic_f:
    for i, line in enumerate(intrinsic_f):
        K[i] = [float(el) for el in line.split()]

# camera intrinsic parameter matrix
D = np.empty(shape=(1, 2), dtype=np.float64)

with open("./data/D.txt", 'r') as intrinsic_f:
    for i, line in enumerate(intrinsic_f):
        D[i] = [float(el) for el in line.split()]

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

img = cv2.imread("./data/images/img_0001.jpg", cv2.IMREAD_GRAYSCALE)

img_undist = np.empty_like(img)

for x in range(img.shape[1]):
    for y in range(img.shape[0]):
        coords = distortPoints(np.array([x, y], dtype=float), D, K)
        coords = np.floor(coords).astype(int)
        if coords[0] >= 0 and coords[0] < img_undist.shape[1] and coords[1] >= 0 and coords[1] < img_undist.shape[0]:
            img_undist[y, x] = img[coords[1]-1, coords[0]-1]

img = img_undist

for pose in poses:
    matrix = poseVectorToTransformationMatrix(pose)
    pixel_coords = projectPoint(corners, matrix, K)

    for i in range(pixel_coords.shape[1]):
        img = cv2.circle(img, (int(pixel_coords[0, i]), int(pixel_coords[1, i])),
                         radius=3, color=(0, 0, 255), thickness=-1)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

