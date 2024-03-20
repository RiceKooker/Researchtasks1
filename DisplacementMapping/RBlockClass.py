from BlockClass import Block
import numpy as np
import Func
from Blender.const import face_vertex_index
import matplotlib.pyplot as plt


def procrustes_analysis(X, Y):
    X = np.array(X)
    Y = np.array(Y)
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
    if len(X[0]) == 2:  # 2D case
        S = np.eye(2)
        S[1, 1] = np.linalg.det(np.dot(Vt.T, U.T))
    else:  # 3D case
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


def find_absolute_loc(block, point_coord):
    centroid = block.get_com()
    rot_angles = block.find_rot()
    # rotated_point = Func.rotate_3d_point(point_coord, rot_angles)
    rotated_point = np.dot(block.find_rot(), point_coord)
    return np.array([i for i in rotated_point + np.array(centroid)])


def generate_points_on_cuboid(vertices, num_points_per_dim):
    """
    Generate points equally spaced on the surfaces of a cuboid.

    Parameters:
    - vertices: List of 8 vertices of the cuboid.
    - num_points_per_dim: Number of points to generate along each dimension.

    Returns:
    - points: List of generated points on the cuboid surfaces.
    """

    # Convert vertices to a NumPy array for easier manipulation
    vertices = np.array(vertices)

    # Get the minimum and maximum coordinates along each axis
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    # Generate points on the surfaces
    points = []

    for axis in range(3):  # Iterate over X, Y, Z axes
        other_axes = [i for i in range(3) if i != axis]

        for coord in np.linspace(min_coords[axis], max_coords[axis], num_points_per_dim):
            # Generate points on the face perpendicular to the current axis
            for point in np.ndindex(num_points_per_dim, num_points_per_dim):
                # Create a point by fixing the current axis and varying the other two axes
                new_point = np.zeros(3)
                new_point[axis] = coord
                for i, other_axis in enumerate(other_axes):
                    new_point[other_axis] = min_coords[other_axis] + (max_coords[other_axis] - min_coords[other_axis]) * point[i] / (num_points_per_dim - 1)
                points.append(new_point)

    return np.array(points)


def rectangle_plane_equation(vertices):
    # Convert vertices to numpy array for easier manipulation
    vertices = np.array(vertices)

    # Define vectors lying in the plane of the rectangle
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]

    # Calculate the normal vector to the plane
    normal_vector = np.cross(v1, v2)

    # Calculate the constant D in the plane equation
    D = -np.dot(normal_vector, vertices[0])

    # Normalize the normal vector for a cleaner representation of the plane equation
    normal_vector /= np.linalg.norm(normal_vector)

    # Construct the plane equation coefficients
    A, B, C = normal_vector

    x_values, y_values, z_values = zip(*vertices)
    max_values = [max(x_values), max(y_values), max(z_values)]
    min_values = [min(x_values), min(y_values), min(z_values)]

    return [A, B, C, D], [max_values, min_values]


def generate_points_on_rectangle(rectangle_vertices, num_points_per_dim):
    vertices = np.array(rectangle_vertices)

    # Define two vectors lying in the plane of the rectangle
    v1 = vertices[1] - vertices[0]
    v2 = vertices[-1] - vertices[0]

    # Create parametric coordinates
    u_values = np.linspace(0, 1, num_points_per_dim)
    v_values = np.linspace(0, 1, num_points_per_dim)

    # Generate points on the surface of the rectangle
    points = []
    for u in u_values:
        for v in v_values:
            point = vertices[0] + u * v1 + v * v2
            points.append(point)

    return np.array(points)


def get_face_verts(face_index, vertices):
    """
    This function groups all the vertices that belong to a face.
    """
    vertices = np.array(vertices)
    temp = []
    for i in face_vertex_index[face_index]:
        temp.append(vertices[i-1])
    return np.array(temp)



class RBlock(Block):
    def __int__(self, x_lim, y_lim, z_lim):
        super().__init__(x_lim, y_lim, z_lim)

    def initial_loc(self):
        vertices = Func.find_vertices([0.5*i for i in self.dims], self.dims)
        return vertices

    def find_rot(self, angle=False):
        initial_v = self.initial_loc()
        rotation_matrix, trans_matrix = procrustes_analysis(self.vertices, initial_v)
        rotation_angles = rotation_matrix_to_euler_angles(rotation_matrix)
        if angle is True:
            return rotation_angles
        else:
            return rotation_matrix

    def draw(self, highlight_point=None, show=True):
        Func.draw_blocks3([self], highlight_point=highlight_point, show=show)

    def find_absolute_pos(self, point_coord):
        """
        This function finds the absolute coordinate of any point given its relative position in the block's principle
        coordinate system.
        :param point_coord: relative position from the centroid of the block, in the block's principle coordinates.
        """
        centroid = self.get_com()
        rotated_point = np.dot(self.find_rot(), point_coord)
        return np.array([i for i in rotated_point + np.array(centroid)])

    def find_relative_pos(self, point_coord):
        point_coord = np.array(point_coord)
        relative_pos = point_coord - self.get_com()
        relative_pos = np.dot(self.find_rot().T, relative_pos)
        return relative_pos

    def generate_surface_points(self, num_points_per_dim, face_i=None):
        if face_i is not None:
            return generate_points_on_rectangle(get_face_verts(face_i, self.vertices), num_points_per_dim)
        surface_points = generate_points_on_rectangle(get_face_verts(0, self.vertices), num_points_per_dim)
        for i in range(1, 6):
            surface_points_ = generate_points_on_rectangle(get_face_verts(i, self.vertices), num_points_per_dim)
            surface_points = np.concatenate((surface_points, surface_points_), axis=0)
        return surface_points



if __name__ == '__main__':
    sample_block1 = RBlock.dim_build([1, 1, 1])
    sample_block1.move([2, 4, 0])
    sample_block1.rot([30, 45, -15])
    abs_coord = sample_block1.find_absolute_pos([0.5, -0.4, -0.5])
    print('Absolute coord: ', abs_coord)
    rel_coord = sample_block1.find_relative_pos(abs_coord)
    print('Relative coord: ', rel_coord)
    # sample_block1.draw(highlight_point=new_coord, show=False)
    # plt.show()
