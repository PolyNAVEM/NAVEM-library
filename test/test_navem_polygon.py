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

import unittest
from NAVEM.geometry.geometry_utilities import *
import matplotlib.pyplot as plt
from matplotlib import use
import importlib
from NAVEM.Utilities.NAVEMPolygon import NAVEMPolygon
from pypolydim import polydim

try:
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True

plot_mapped = True

def plot_mapped_polygon(vertex_points: List[NDArray[np.float64]], inertia_vertex_points: List[NDArray[np.float64]], mapped_vertex_points: List[NDArray[np.float64]]) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    ax = axes[0]
    i = 1
    for vertex in vertex_points:
        num_vertices = vertex.shape[1]
        coord = np.zeros([3, num_vertices + 1])
        coord[:, 0:num_vertices] = vertex
        coord[:, num_vertices] = vertex[:, 0]
        ax.plot(coord[0, :], coord[1, :], label='Polygon' + str(i), linewidth=3.0, markersize=12)
        i = i + 1

    ax.title.set_text('original polygon')

    ax = axes[1]
    i = 1
    for vertex in inertia_vertex_points:
        num_vertices = vertex.shape[1]
        coord = np.zeros([3, num_vertices + 1])
        coord[:, 0:num_vertices] = vertex
        coord[:, num_vertices] = vertex[:, 0]
        ax.plot(coord[0, :], coord[1, :], label='Polygon' + str(i), linewidth=3.0, markersize=12)
        i = i + 1

    ax.title.set_text('inertia polygon')


    ax = axes[2]
    i = 1
    for vertex in mapped_vertex_points:
        num_vertices = vertex.shape[1]
        coord = np.zeros([3, num_vertices + 1])
        coord[:, 0:num_vertices] = vertex
        coord[:, num_vertices] = vertex[:, 0]
        ax.plot(coord[0, :], coord[1, :], label='Polygon' + str(i), linewidth=3.0, markersize=12)
        i = i + 1

    ax.title.set_text('mapped polygon')

    plt.show()

class TestPolygonUtilities(unittest.TestCase):

    def test_navem_polygon_on_concave(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        num_vertices = 4
        vertex_points = np.zeros([3, num_vertices])
        vertex_points[0, :] = [0.0, 0.5, 0.5, 0.35]
        vertex_points[1, :] = [0.0, 0.0, 0.5, 0.15]

        area = geometry_utilities.polygon_area(vertex_points)
        centroid = geometry_utilities.polygon_centroid(vertex_points, area)
        diameter = geometry_utilities.polygon_diameter(vertex_points)
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertex_points, centroid)

        polygon = NAVEMPolygon(geometry_utilities, vertex_points, vertex_points, centroid, diameter, list_triangles)
        inertia_mapped_points, _ = polygon.map_inertia_inv(vertex_points)
        _, mapped_vertices, _, _= polygon.map_f_inv(vertex_points, 0)

        if plot_mapped:
            plot_mapped_polygon( [vertex_points, polygon.kernel],
                                 [inertia_mapped_points],
                                 [mapped_vertices])

    def test_eig_matrix(self):
        mass_matrix = np.zeros([3, 3])
        mass_matrix[0, :] = [2.0, 1.0, 1.0]
        mass_matrix[1, :] = [1.0, 2.0, 1.0]
        mass_matrix[2, :] = [1.0, 1.0, 2.0]

        eigenvalues, eigenvectors = np.linalg.eigh(mass_matrix)
        u, s, vh = np.linalg.svd(mass_matrix)

        self.assertLess(np.linalg.norm(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T - mass_matrix), 1.0e-12)
        self.assertLess(np.linalg.norm(u @ np.diag(s) @ vh - mass_matrix), 1.0e-12)



    def test_inertia_mapping_on_parallelograms(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        # First polygon
        num_vertices = 4
        vertices = np.zeros([3, num_vertices])
        vertices[0, :] = [0.0, 1.0, 0.8, 0.4]
        vertices[1, :] = [0.0, 0.0, 1.0, 0.9]


        area = geometry_utilities.polygon_area(vertices)
        centroid = geometry_utilities.polygon_centroid(vertices, area)
        diameter = geometry_utilities.polygon_diameter(vertices)
        inv_diameter = 1.0 / diameter
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertices, centroid)

        # first re-scaling
        fr_vertex_points = inv_diameter * vertices

        # first inertia mapping
        fr_centroid = inv_diameter * centroid
        list_triangles_points = geometry_utilities.extract_triangulation_points_by_internal_point(fr_vertex_points, fr_centroid, list_triangles)
        gedim_mass_matrix = geometry_utilities.polygon_mass(centroid, list_triangles_points)

        monomials = polydim.utilities.Monomials_2D()
        monomials_data = monomials.compute(1)
        quadrature = gedim.quadrature.Quadrature_Gauss2D_Triangle()
        quadrature_data = quadrature.fill_points_and_weights(2)

        vem_quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
        vem_quadrature_data = vem_quadrature.polygon_internal_quadrature(quadrature_data, list_triangles_points)
        internal_nodes = vem_quadrature_data.points
        internal_weights = vem_quadrature_data.weights
        vandermonde = monomials.vander(monomials_data, internal_nodes, centroid, 1.0)
        internal_weights_sqrt = np.sqrt(internal_weights)
        temp = np.diag(internal_weights_sqrt) @ vandermonde
        mass_matrix = (temp.T @ temp)[1:, 1:]

        temp_1 = np.diag(internal_weights_sqrt) @ vandermonde[:, 1:]
        mass_matrix_1 = (temp_1.T @ temp_1)

        # C = U @ S @ VH
        u, s, vh = np.linalg.svd(temp_1)

        # M = V @ D @ V.T = C.T @ C = VH.T @ S @ U.T @ U @ S @ VH = VH.T @ S @ S @ VH
        eigenvalues, eigenvectors = np.linalg.eigh(mass_matrix)

        b_matrix = np.identity(3)
        b_matrix[0:2, 0:2] = np.sqrt(max(eigenvalues)) * np.sqrt(np.diag(np.reciprocal(eigenvalues))) @ eigenvectors.T

        eigenvalues_1 = s**2
        eigenvectors_1 = vh.T
        b_matrix_1 = np.identity(3)
        b_matrix_1[0:2, 0:2] = np.sqrt(max(eigenvalues_1)) * np.sqrt(np.diag(np.reciprocal(eigenvalues_1))) @ eigenvectors_1.T


        self.assertLess(np.linalg.norm(gedim_mass_matrix - mass_matrix), 1.0e-12)
        self.assertLess(np.linalg.norm(gedim_mass_matrix - mass_matrix_1), 1.0e-12)
        self.assertLess(np.linalg.norm(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T - mass_matrix), 1.0e-12)
        self.assertLess(np.linalg.norm(vh.T @ np.diag(s) @ np.diag(s) @ vh - mass_matrix), 1.0e-12)


    def test_polygon_mapping_on_parallelograms(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        # First polygon
        num_vertices = 4
        vertex_points = np.zeros([3, num_vertices])
        vertex_points[0, :] = [1.4, 1.9, 2.4, 1.9]
        vertex_points[1, :] = [0.035, 0.025, 0.035, 0.045]


        area = geometry_utilities.polygon_area(vertex_points)
        centroid = geometry_utilities.polygon_centroid(vertex_points, area)
        diameter = geometry_utilities.polygon_diameter(vertex_points)
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertex_points, centroid)

        polygon = NAVEMPolygon(geometry_utilities, vertex_points, vertex_points, centroid, diameter, list_triangles)

        inertia_mapped_points, _ = polygon.map_inertia_inv(vertex_points)
        _, mapped_vertices, _, _ = polygon.map_f_inv(vertex_points, 0)

        # Second polygon
        other_vertex_points = np.zeros([3, num_vertices])
        other_vertex_points[0, :] = [0.0, 1.0, 1.2, 0.2]
        other_vertex_points[1, :] = [0.02, 0.02, 0.05, 0.05]

        other_area = geometry_utilities.polygon_area(other_vertex_points)
        other_centroid = geometry_utilities.polygon_centroid(other_vertex_points, other_area)
        other_diameter = geometry_utilities.polygon_diameter(other_vertex_points)
        other_list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(other_vertex_points, other_centroid)

        other_polygon = NAVEMPolygon(geometry_utilities, other_vertex_points, other_vertex_points, other_centroid, other_diameter, other_list_triangles)

        other_inertia_mapped_points, _ = other_polygon.map_inertia_inv(other_vertex_points)
        _, other_mapped_vertices, _, _ = other_polygon.map_f_inv(other_vertex_points, 0)

        if plot_mapped:
            plot_mapped_polygon( [vertex_points, other_vertex_points],
                                 [inertia_mapped_points, other_inertia_mapped_points],
                                 [mapped_vertices, other_mapped_vertices])

        self.assertLess(np.linalg.norm(mapped_vertices - other_mapped_vertices), 1.0e-12)


    def test_polygon_mapping_on_parallelograms_2(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        num_vertices = 4

        # First polygon
        vertex_points = np.zeros([3, num_vertices])
        vertex_points[0, :] = [-3.2931564766475002e-01, 8.2904083598123200e-10, 8.2904083598123200e-10, -3.2881217384530997e-01]
        vertex_points[1, :] = [-3.2683371802085198e-01, -3.2970894474863799e-01, -8.3629081437663899e-10, -8.3629081437663899e-10]

        area = geometry_utilities.polygon_area(vertex_points)
        centroid = geometry_utilities.polygon_centroid(vertex_points, area)
        diameter = geometry_utilities.polygon_diameter(vertex_points)
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertex_points, centroid)

        polygon = NAVEMPolygon(geometry_utilities, vertex_points, vertex_points, centroid, diameter, list_triangles)

        inertia_mapped_points, _ = polygon.map_inertia_inv(vertex_points)
        _, mapped_vertices, _, _ = polygon.map_f_inv(vertex_points, 0)

        # Second Polygon
        other_vertex_points = np.zeros([3, num_vertices])
        other_vertex_points[0, :] = [-3.29e-01, 8.29e-10, 8.29e-10, -3.29e-01]
        other_vertex_points[1, :] = [-3.29e-01, -3.29e-01, 8.29e-10, 8.29e-10]

        other_area = geometry_utilities.polygon_area(other_vertex_points)
        other_centroid = geometry_utilities.polygon_centroid(other_vertex_points, other_area)
        other_diameter = geometry_utilities.polygon_diameter(other_vertex_points)
        other_list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(other_vertex_points, other_centroid)

        other_polygon = NAVEMPolygon(geometry_utilities, other_vertex_points, other_vertex_points, other_centroid, other_diameter, other_list_triangles)

        other_inertia_mapped_points, _ = other_polygon.map_inertia_inv(other_vertex_points)
        _, other_mapped_vertices, _, _ = other_polygon.map_f_inv(other_vertex_points, 0)

        if plot_mapped:
            plot_mapped_polygon([vertex_points, other_vertex_points],
                                [inertia_mapped_points, other_inertia_mapped_points],
                                [mapped_vertices, other_mapped_vertices])

    def test_polygon_mapping_on_concave(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        # Polygon mapped as standard
        num_vertices = 4
        vertex_points = np.zeros([3, num_vertices])
        # vertex_points[0, :] = [9.91463, 1.93238, -12.832, 1]
        # vertex_points[1, :] = [-14.0879, 11.3091, -9.16793, 0]

        vertex_points[0, :] = [1., 0.13868399, 0., 0.30197903]
        vertex_points[1, :] = [1., 0.72279196, 0.90146038, 0.]

        area = geometry_utilities.polygon_area(vertex_points)
        centroid = geometry_utilities.polygon_centroid(vertex_points, area)
        polygon_internal_angles = compute_polygon_interior_angles(vertex_points)
        kernel = compute_polygon_kernel(geometry_utilities, vertex_points, polygon_internal_angles)
        diameter = geometry_utilities.polygon_diameter(kernel)
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertex_points, centroid)

        polygon = NAVEMPolygon(geometry_utilities, vertex_points, vertex_points, centroid, diameter, list_triangles)

        inertia_mapped_points, _ = polygon.map_inertia_inv(kernel)
        _, mapped_vertices, _, _ = polygon.map_f_inv(kernel, 3)

        inertia_mapped_kernel, _ = polygon.map_inertia_inv(kernel)
        _, mapped_kernel, _, _ = polygon.map_f_inv(kernel, 3)


        # Polygon mapped by applying the maps to its kernel
        kernel_area = geometry_utilities.polygon_area(kernel)

        if kernel_area < geometry_utilities.tolerance2_d() or kernel_area < area * 0.2:
            raise ValueError("Not valid polygon")

        kernel_centroid = geometry_utilities.polygon_centroid(kernel, kernel_area)
        other_centroid = kernel_centroid
        other_diameter = geometry_utilities.polygon_diameter(kernel)
        other_list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(kernel, other_centroid)

        other_polygon = NAVEMPolygon(geometry_utilities, vertex_points, kernel, other_centroid, other_diameter, other_list_triangles)

        other_inertia_mapped_points, _ = other_polygon.map_inertia_inv(vertex_points)
        _, other_mapped_vertices, _, _ = other_polygon.map_f_inv(vertex_points, 3)

        other_inertia_mapped_kernel, _ = other_polygon.map_inertia_inv(kernel)
        _, other_mapped_kernel, _, _ = other_polygon.map_f_inv(kernel, 3)

        if plot_mapped:
            plot_mapped_polygon([vertex_points, vertex_points, kernel, kernel],
                                [inertia_mapped_points, other_inertia_mapped_points, inertia_mapped_kernel, other_inertia_mapped_kernel],
                                [mapped_vertices, other_mapped_vertices, mapped_kernel, other_mapped_kernel])


if __name__ == '__main__':

    unittest.main()
