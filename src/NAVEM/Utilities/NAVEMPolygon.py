# _LICENSE_HEADER_
#
# Copyright (C) 2026.
# Terms register on the GPL-3.0 license.
#
# This file can be redistributed and/or modified under the license terms.
#
# See top level LICENSE file for more details.
#
# This file can be used citing references in CITATION.cff file.

import numpy as np
from pypolydim import gedim
from numpy.typing import NDArray
from typing import Tuple, List

# Map the points to the similar polygon with centroid=(0,0) and diameter=1
class NAVEMPolygon:

    def __init__(self, geometry_utilities: gedim.GeometryUtilities, vertex_points: NDArray[np.float64], diameter: float,
                 centroid: NDArray[np.float64], list_triangles: List[int], use_kernel: bool = False, kernel: NDArray[np.float64] = None):

        self.geometry_utilities = geometry_utilities
        self.list_triangles = list_triangles
        self.diameter = diameter
        self.inv_diameter = 1.0 / diameter
        self.centroid = np.expand_dims(centroid, axis=1)
        self.num_vertices: int = vertex_points.shape[1]

        if use_kernel:
            num_vertices: int = vertex_points.shape[1]
            num_kernel_vertices: int = kernel.shape[1]

            self.vertices = kernel
            self.mapped_max_vertex_distance = []

            # first re-scaling
            fr_vertex_points = self.inv_diameter * self.vertices

            # inertia mapping
            fr_centroid = self.inv_diameter * self.centroid
            in_vertex_points, map_inertia_matrix = self.compute_map_inertia(fr_vertex_points, fr_centroid, self.list_triangles)

            # second inertia mapping for stability reason
            in_centroid = np.zeros([3, 1])
            s_in_vertex_points, s_map_inertia_matrix = self.compute_map_inertia(in_vertex_points, in_centroid, self.list_triangles)

            # second rescaling
            s_in_diameter = self.geometry_utilities.polygon_diameter(s_in_vertex_points)
            s_in_inv_diameter = 1.0 / s_in_diameter

            sr_vertex_points = np.zeros([3, num_kernel_vertices])
            for v in range(num_kernel_vertices):
                sr_vertex_points[:, v:(v + 1)] = s_in_inv_diameter * s_in_vertex_points[:, v:(v + 1)]

            self.map_f_matrix = self.inv_diameter * s_in_inv_diameter * s_map_inertia_matrix @ map_inertia_matrix

            self.mapped_vertices = np.zeros([3, num_vertices])
            for v in range(num_vertices):
                self.mapped_vertices[:, v:(v + 1)] = (self.map_f_matrix @
                                                    (vertex_points[:, v:(v + 1)] - self.centroid))

        else:

            self.vertices = vertex_points
            self.mapped_max_vertex_distance = []

            # first re-scaling
            fr_vertex_points = self.inv_diameter * self.vertices

            # inertia mapping
            fr_centroid = self.inv_diameter * self.centroid
            in_vertex_points, map_inertia_matrix = self.compute_map_inertia(fr_vertex_points, fr_centroid, self.list_triangles)

            # second inertia mapping for stability reason
            in_centroid = np.zeros([3, 1])
            s_in_vertex_points, s_map_inertia_matrix = self.compute_map_inertia(in_vertex_points, in_centroid, self.list_triangles)

            # second rescaling
            s_in_diameter = self.geometry_utilities.polygon_diameter(s_in_vertex_points)
            s_in_inv_diameter = 1.0 / s_in_diameter

            sr_vertex_points = np.zeros([3, self.num_vertices])
            for v in range(self.num_vertices):
                sr_vertex_points[:, v:(v + 1)] = s_in_inv_diameter * s_in_vertex_points[:, v:(v + 1)]

            self.mapped_vertices = sr_vertex_points
            self.map_f_matrix = self.inv_diameter * s_in_inv_diameter * s_map_inertia_matrix @ map_inertia_matrix


    def compute_inertia_matrix(self, points: NDArray[np.float64], centroid: NDArray[np.float64], list_triangles: List[int]) -> NDArray[np.float64]:

        list_triangles_points = self.geometry_utilities.extract_triangulation_points(points, list_triangles)
        mass_matrix = self.geometry_utilities.polygon_mass(np.squeeze(centroid, axis=1), list_triangles_points)

        return mass_matrix

    def compute_map_inertia(self, points: NDArray[np.float64], centroid: NDArray[np.float64], list_triangles: List[int]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        num_vertices: int = points.shape[1]

        # inertia mapping
        mass_matrix = self.compute_inertia_matrix(points, centroid, list_triangles)
        eigenvalues, eigenvectors = np.linalg.eigh(mass_matrix)
        det_q = np.linalg.det(eigenvectors)
        b_matrix = np.identity(3)
        b_matrix[0:2, 0:2] = np.sqrt(max(eigenvalues)) * np.sqrt(np.diag(np.reciprocal(eigenvalues))) @ eigenvectors.T

        flip_matrix = np.identity(3)
        if det_q < 0.0:
            flip_matrix[1, 1] = -1

        mapped_points = np.zeros([3, num_vertices])
        for v in range(num_vertices):
            mapped_points[:, v:(v + 1)] = flip_matrix @ b_matrix @ (points[:, v:(v + 1)] - centroid[:, 0:1])

        map_inertia_matrix = flip_matrix @ b_matrix

        return mapped_points, map_inertia_matrix

    def map_f_inv(self, points: NDArray[np.float64], index: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:

        vertex = self.mapped_vertices[:, index]
        norm_first_vertex = np.linalg.norm(vertex)

        if vertex[0] >= 0:
            if vertex[1] >= 0:
                theta = 2 * np.pi - np.arccos(vertex[0] / norm_first_vertex)
            else:
                theta = np.arccos(vertex[0] / norm_first_vertex)
        elif vertex[0] <= 0:
            if vertex[1] >= 0:
                theta = np.arccos(- vertex[0] / norm_first_vertex) + np.pi
            else:
                theta = np.pi - np.arccos(-vertex[0] / norm_first_vertex)
        else:
            raise ValueError("Not observed case")


        # Compute cos and sin once
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                    [sin_theta, cos_theta, 0.0],
                                    [0.0, 0.0, 1.0]])

        if np.linalg.det(rotation_matrix) < 0:
            raise ValueError("Rotation matrix has a negative determinant.")

        rotated_vertices = rotation_matrix @ self.mapped_vertices

        if rotated_vertices[0, index] <= 0:
            raise ValueError("Negative x-coordinate encountered after rotation.")

        rescaling_factor = rotated_vertices[0, index]
        inv_rescaling_factor = (1 / rescaling_factor)

        resc_rotated_vertex = inv_rescaling_factor * rotated_vertices

        # Vectorized mapping
        mapped_points = inv_rescaling_factor * rotation_matrix @ self.map_f_matrix @ (points - self.centroid)

        jac_inv = inv_rescaling_factor * rotation_matrix @ self.map_f_matrix

        return resc_rotated_vertex, mapped_points, jac_inv, float(inv_rescaling_factor)

    def map_inertia_inv(self, points: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        num_points = points.shape[1]
        mapped_points = np.zeros([3, num_points])
        for v in range(num_points):
            mapped_points[:, v:(v + 1)] = (self.map_f_matrix @
                                           (points[:, v:(v + 1)] - self.centroid))

        return mapped_points, self.map_f_matrix

    def __str__(self):
        return str(self.diameter) + str(self.centroid)

    @staticmethod
    def map_fix_vertex_inv(mapped_vertices: NDArray[np.float64], inertia_points: NDArray[np.float64], index: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        vertex = mapped_vertices[:, index]
        norm_first_vertex = np.linalg.norm(vertex)

        if vertex[0] >= 0:
            if vertex[1] >= 0:
                theta = 2 * np.pi - np.arccos(vertex[0] / norm_first_vertex)
            else:
                theta = np.arccos(vertex[0] / norm_first_vertex)
        elif vertex[0] <= 0:
            if vertex[1] >= 0:
                theta = np.arccos(- vertex[0] / norm_first_vertex) + np.pi
            else:
                theta = np.pi - np.arccos(-vertex[0] / norm_first_vertex)
        else:
            raise ValueError("Not observed case")


        # Compute cos and sin once
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                    [sin_theta, cos_theta, 0.0],
                                    [0.0, 0.0, 1.0]])

        if np.linalg.det(rotation_matrix) < 0:
            raise ValueError("Rotation matrix has a negative determinant.")

        rotated_vertices = rotation_matrix @ mapped_vertices

        if rotated_vertices[0, index] <= 0:
            raise ValueError("Negative x-coordinate encountered after rotation.")

        rescaling_factor = rotated_vertices[0, index]
        inv_rescaling_factor = (1 / rescaling_factor)

        resc_rotated_vertex = inv_rescaling_factor * rotated_vertices

        # Vectorized mapping
        mapped_points = inv_rescaling_factor * rotation_matrix @ inertia_points

        jac_inv = inv_rescaling_factor * rotation_matrix
        jac = rescaling_factor * rotation_matrix.T

        return resc_rotated_vertex, mapped_points, jac_inv, jac, float(inv_rescaling_factor)


