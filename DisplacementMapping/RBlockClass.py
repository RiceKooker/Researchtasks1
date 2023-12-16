import sys
from BlockClass import Block
import numpy as np
from scipy.linalg import eig, norm
import Func
from scipy.linalg import svd
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


def procrustes_analysis(X, Y):
    # Center the points
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Compute the cross-covariance matrix
    covariance_matrix = np.dot(X_centered.T, Y_centered)

    # Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Ensure a proper rotation matrix (fix potential reflections)
    S = np.eye(3)
    S[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))

    # Compute the rotation matrix
    rotation_matrix = np.dot(U, np.dot(S, Vt))

    # Calculate the translation vector
    translation_vector = centroid_X - np.dot(rotation_matrix, centroid_Y)

    return rotation_matrix, translation_vector


def convert_vertex_order(vertices):
    temp = np.array([i for i in vertices])
    temp[1] = vertices[3]
    temp[2] = vertices[7]
    temp[3] = vertices[4]
    temp[4] = vertices[1]
    temp[5] = vertices[2]
    temp[7] = vertices[5]
    return temp


def rotation_matrix_to_euler_angles(rotation_matrix):
    # Extract individual elements from the rotation matrix
    r00, r01, r02 = rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2]
    r10, r11, r12 = rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2]
    r20, r21, r22 = rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2]

    # Calculate the pitch (rotation around Y-axis)
    pitch = np.arctan2(-r20, np.sqrt(r21**2 + r22**2))

    # Calculate the yaw (rotation around Z-axis)
    yaw = np.arctan2(r10, r00)

    # Calculate the roll (rotation around X-axis)
    roll = np.arctan2(r21, r22)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)



class RBlock(Block):
    def __int__(self, x_lim, y_lim, z_lim):
        super().__init__(x_lim, y_lim, z_lim)

    def initial_loc(self):
        vertices = Func.find_vertices([0.5*i for i in self.dims], self.dims)
        return vertices

    def find_rot(self):
        initial_v = self.initial_loc()
        rotation_matrix, trans_matrix = procrustes_analysis(self.vertices, initial_v)

        rotation_angles = rotation_matrix_to_euler_angles(rotation_matrix)

        return rotation_angles


if __name__ == '__main__':
    sample_block1 = RBlock.dim_build([1,2,3])
    sample_block1.rot([45, 23, 67])
    final = np.array([i for i in sample_block1.vertices])
    print(sample_block1.find_rot())
