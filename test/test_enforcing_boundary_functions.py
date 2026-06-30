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
from NAVEM.Utilities.enforcing_boundary_functions import EnforcingBoundary, BoundaryMethodType, BubbleType, lagrange_1d_values
import tensorflow as tf
import numpy as np
from pypolydim import gedim, polydim
import matplotlib.pyplot as plt
from matplotlib import use
import importlib
from NAVEM.Utilities.points_generator import chebyshev_lobatto_nodes
from NAVEM.Utilities.NAVEMPolygon import NAVEMPolygon

try:
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True


def plot_polygon_with_edge_normals(vertices: np.ndarray,
                                   edge_normals: np.ndarray,
                                   normal_length: float = 0.15) -> None:

    num_vertices = vertices.shape[1]
    next_vertices = np.roll(vertices, shift=-1, axis=1)

    edge_mid_points = 0.5 * (vertices + next_vertices)


    fig, ax = plt.subplots(figsize=(6, 6))

    polygon_closed = np.zeros([3, vertices.shape[1]+1])
    polygon_closed[:, :vertices.shape[1]] = vertices
    polygon_closed[:, vertices.shape[1]] = vertices[:, 0]
    ax.plot(polygon_closed[0, :], polygon_closed[1, :], 'b-', linewidth=2, zorder=1)
    ax.scatter(vertices[0, :], vertices[1, :], c='blue', s=50, zorder=3, label='polygon')

    ax.scatter(edge_mid_points[0, :], edge_mid_points[1, :], c='black', s=20, zorder=3)

    for i in range(num_vertices):
        mp = edge_mid_points[:, i]
        n = edge_normals[:, i]
        ax.annotate(
            '', xy=(mp[0] + normal_length * n[0], mp[1] + normal_length * n[1]),
            xytext=(mp[0], mp[1]),
            arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
            zorder=2
        )

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


class TestEnforcingBoundaryFunctionsEfficient(unittest.TestCase):


    def test_laplacian_values_g(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
        reference_quadrature = gedim.quadrature.Quadrature_Gauss1D()
        reference_segment_quadrature_data = reference_quadrature.fill_points_and_weights(20)
        reference_triangle_quadrature_data: gedim.quadrature.QuadratureData \
            = gedim.quadrature.Quadrature_Gauss2D_Triangle.fill_points_and_weights(20)

        polygon_vertices = np.array([[0.90, 0.65, -0.35, -0.90, 0.10],
                              [-0.80, 0.86, 0.60, 0.10, -0.60],
                              [0.0, 0.0, 0.0, 0.0, 0.0]])

        polygon_edge_lengths = geometry_utilities.polygon_edge_lengths(polygon_vertices)
        polygon_edge_tangents = geometry_utilities.polygon_edge_tangents(polygon_vertices)
        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        polygon_area = geometry_utilities.polygon_area(polygon_vertices)
        polygon_centroid = geometry_utilities.polygon_centroid(polygon_vertices, polygon_area)
        polygon_list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(polygon_vertices, polygon_centroid)
        polygon_triangulations = geometry_utilities.extract_triangulation_points_by_internal_point(polygon_vertices, polygon_centroid, polygon_list_triangles)

        plot_polygon_with_edge_normals(polygon_vertices, polygon_edge_normals, 0.15)

        boundary_quadrature_data = quadrature.polygon_edges_quadrature(reference_segment_quadrature_data,
                                                              polygon_vertices,
                                                              polygon_edge_lengths,
                                                              [True for _ in range(polygon_vertices.shape[1])],
                                                              polygon_edge_tangents,
                                                              polygon_edge_normals)

        internal_quadrature_data \
            = quadrature.polygon_internal_quadrature(reference_triangle_quadrature_data, polygon_triangulations)


        method_order = 1
        method_type_g = BoundaryMethodType.wachspress
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_g=method_type_g)

        vertices = np.array([polygon_vertices[:2, :]])
        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        eb.initialize_boundary_properties(vertices)

        xy_internal = internal_quadrature_data.points[:2, :].T
        xy_internal = tf.convert_to_tensor(xy_internal, dtype=tf.float64)
        xy_internal = tf.expand_dims(xy_internal, 0)
        xy_internal = tf.concat([xy_internal] * vertices.shape[0], axis=0)

        xy_boundary = boundary_quadrature_data.quadrature.points[:2, :].T
        xy_boundary = tf.convert_to_tensor(xy_boundary, dtype=tf.float64)
        xy_boundary = tf.expand_dims(xy_boundary, 0)
        xy_boundary = tf.concat([xy_boundary] * vertices.shape[0], axis=0)

        g_internal, g_grad_internal, g_sec_ders_internal = eb.compute_g_and_grads_and_second_derivatives(xy_internal)
        g_internal_laplacian = np.array((g_sec_ders_internal[0, :, :, 0] + g_sec_ders_internal[0, :, :, 2]))
        g_boundary, g_grad_boundary = eb.compute_g_and_grads(xy_boundary)
        g_x_boundary = g_grad_boundary[0, :, :, 0]
        g_y_boundary = g_grad_boundary[0, :, :, 1]


        term_a_g = (np.expand_dims(internal_quadrature_data.weights, axis=1).T @ g_internal_laplacian)

        term_b_g = np.zeros(g_y_boundary.shape)
        num_points_on_each_edge = reference_segment_quadrature_data.points.shape[1]

        for i in range(g_internal_laplacian.shape[1]):
            for e in range(polygon_vertices.shape[1]):
                term_b_g[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] \
                    = (g_x_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] *
                       polygon_edge_normals[0, e]
                       + g_y_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] *
                       polygon_edge_normals[1, e])

        term_b_g = (np.expand_dims(boundary_quadrature_data.quadrature.weights, axis=1).T @ term_b_g)

        print("error: ", np.linalg.norm(term_a_g - term_b_g))

    def test_laplacian_values_g_map(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
        reference_quadrature = gedim.quadrature.Quadrature_Gauss1D()
        reference_segment_quadrature_data = reference_quadrature.fill_points_and_weights(20)
        reference_triangle_quadrature_data: gedim.quadrature.QuadratureData \
            = gedim.quadrature.Quadrature_Gauss2D_Triangle.fill_points_and_weights(20)

        polygon_vertices = np.array([[0.90, 0.65, -0.35, -0.90, 0.10],
                              [-0.80, 0.86, 0.60, 0.10, -0.60],
                              [0.0, 0.0, 0.0, 0.0, 0.0]])

        polygon_edge_lengths = geometry_utilities.polygon_edge_lengths(polygon_vertices)
        polygon_edge_tangents = geometry_utilities.polygon_edge_tangents(polygon_vertices)
        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        polygon_area = geometry_utilities.polygon_area(polygon_vertices)
        polygon_centroid = geometry_utilities.polygon_centroid(polygon_vertices, polygon_area)
        polygon_diameter = geometry_utilities.polygon_diameter(polygon_vertices)
        polygon_list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(polygon_vertices, polygon_centroid)
        polygon_triangulations = geometry_utilities.extract_triangulation_points_by_internal_point(polygon_vertices, polygon_centroid, polygon_list_triangles)

        # plot_polygon_with_edge_normals(polygon_vertices, polygon_edge_normals, 0.15)

        boundary_quadrature_data = quadrature.polygon_edges_quadrature(reference_segment_quadrature_data,
                                                                      polygon_vertices,
                                                                      polygon_edge_lengths,
                                                                      [True for _ in range(polygon_vertices.shape[1])],
                                                                      polygon_edge_tangents,
                                                                      polygon_edge_normals)

        internal_quadrature_data \
            = quadrature.polygon_internal_quadrature(reference_triangle_quadrature_data, polygon_triangulations)


        method_order = 1
        method_type_g = BoundaryMethodType.wachspress
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_g=method_type_g)

        navem_polygon = NAVEMPolygon(geometry_utilities, polygon_vertices, polygon_vertices, polygon_centroid, polygon_diameter, polygon_list_triangles)

        vertices = np.array([navem_polygon.mapped_vertices[:2, :]])
        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        eb.initialize_boundary_properties(vertices)

        xy_internal, jac_inertia = navem_polygon.map_inertia_inv(internal_quadrature_data.points)
        xy_internal = xy_internal[:2, :].T
        xy_internal = tf.convert_to_tensor(xy_internal, dtype=tf.float64)
        xy_internal = tf.expand_dims(xy_internal, 0)
        xy_internal = tf.concat([xy_internal] * vertices.shape[0], axis=0)

        xy_boundary, _ = navem_polygon.map_inertia_inv(boundary_quadrature_data.quadrature.points)
        xy_boundary = xy_boundary[:2, :].T
        xy_boundary = tf.convert_to_tensor(xy_boundary, dtype=tf.float64)
        xy_boundary = tf.expand_dims(xy_boundary, 0)
        xy_boundary = tf.concat([xy_boundary] * vertices.shape[0], axis=0)

        jac_per_pol = np.zeros(shape=(1, 2, 2, polygon_vertices.shape[1]))
        for i in range(polygon_vertices.shape[1]):
            jac_per_pol[0, :, :, i] = jac_inertia[:2, :2]
        jac_per_pol = tf.convert_to_tensor(jac_per_pol, dtype=tf.float64)

        g_internal, g_grad_internal, g_sec_ders_internal = eb.compute_g_and_grads_and_second_derivatives(xy_internal)
        g_sec_ders_internal = eb.map_g_second_derivatives(g_sec_ders_internal, jac_per_pol)
        g_internal_laplacian = np.array((g_sec_ders_internal[0, :, :, 0] + g_sec_ders_internal[0, :, :, 2]))
        g_boundary, g_grad_boundary = eb.compute_g_and_grads(xy_boundary)
        g_grad_boundary = eb.map_g_grads(g_grad_boundary, jac_per_pol)
        g_x_boundary = g_grad_boundary[0, :, :, 0]
        g_y_boundary = g_grad_boundary[0, :, :, 1]


        term_a_g = (np.expand_dims(internal_quadrature_data.weights, axis=1).T @ g_internal_laplacian)

        term_b_g = np.zeros(g_y_boundary.shape)
        num_points_on_each_edge = reference_segment_quadrature_data.points.shape[1]

        for i in range(g_internal_laplacian.shape[1]):
            for e in range(polygon_vertices.shape[1]):
                term_b_g[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] \
                    = (g_x_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] *
                       polygon_edge_normals[0, e]
                       + g_y_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] *
                       polygon_edge_normals[1, e])

        term_b_g = (np.expand_dims(boundary_quadrature_data.quadrature.weights, axis=1).T @ term_b_g)

        print("error: ", np.linalg.norm(term_a_g - term_b_g))

    def test_laplacian_values_phi(self):

        """
        No-regularity of segment and adfs change sing to integrals
        """

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
        reference_quadrature = gedim.quadrature.Quadrature_Gauss1D()
        reference_segment_quadrature_data = reference_quadrature.fill_points_and_weights(20)
        reference_triangle_quadrature_data: gedim.quadrature.QuadratureData \
            = gedim.quadrature.Quadrature_Gauss2D_Triangle.fill_points_and_weights(20)

        polygon_vertices = np.array([[0.90, 0.65, -0.35, -0.90, 0.10],
                              [-0.80, 0.86, 0.60, 0.10, -0.60],
                              [0.0, 0.0, 0.0, 0.0, 0.0]])

        polygon_edge_lengths = geometry_utilities.polygon_edge_lengths(polygon_vertices)
        polygon_edge_tangents = geometry_utilities.polygon_edge_tangents(polygon_vertices)
        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        polygon_area = geometry_utilities.polygon_area(polygon_vertices)
        polygon_centroid = geometry_utilities.polygon_centroid(polygon_vertices, polygon_area)
        polygon_list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(polygon_vertices, polygon_centroid)
        polygon_triangulations = geometry_utilities.extract_triangulation_points_by_internal_point(polygon_vertices, polygon_centroid, polygon_list_triangles)

        plot_polygon_with_edge_normals(polygon_vertices, polygon_edge_normals, 0.15)

        boundary_quadrature_data = quadrature.polygon_edges_quadrature(reference_segment_quadrature_data,
                                                              polygon_vertices,
                                                              polygon_edge_lengths,
                                                              [True for _ in range(polygon_vertices.shape[1])],
                                                              polygon_edge_tangents,
                                                              polygon_edge_normals)

        internal_quadrature_data \
            = quadrature.polygon_internal_quadrature(reference_triangle_quadrature_data, polygon_triangulations)


        method_order = 1
        method_type_adf = BoundaryMethodType.line
        bubble_type = BubbleType.product
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_adfs=method_type_adf,
                               bubble_type=bubble_type)

        vertices = np.array([polygon_vertices[:2, :]])
        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        eb.initialize_boundary_properties(vertices)

        xy_internal = internal_quadrature_data.points[:2, :].T
        xy_internal = tf.convert_to_tensor(xy_internal, dtype=tf.float64)
        xy_internal = tf.expand_dims(xy_internal, 0)
        xy_internal = tf.concat([xy_internal] * vertices.shape[0], axis=0)

        num_points_on_each_edge = reference_segment_quadrature_data.points.shape[1]
        xy_boundary = np.array(boundary_quadrature_data.quadrature.points[:2, :].T)
        for e in range(polygon_vertices.shape[1]):
            xy_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, 0] \
                = xy_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, 0] - 1.0e-10 * polygon_edge_normals[0, e]
            xy_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, 1] \
                = xy_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, 1] - 1.0e-10 * polygon_edge_normals[1, e]

        xy_boundary = tf.convert_to_tensor(xy_boundary, dtype=tf.float64)
        xy_boundary = tf.expand_dims(xy_boundary, 0)
        xy_boundary = tf.concat([xy_boundary] * vertices.shape[0], axis=0)

        phi_internal, phi_grad_internal, phi_sec_ders_internal = eb.compute_adfs_and_grads_and_second_derivatives(xy_internal)
        phi_internal_laplacian = np.expand_dims((phi_sec_ders_internal[0, :, 0] + phi_sec_ders_internal[0, :, 2]).numpy(), axis=1)
        phi_boundary, phi_grad_boundary = eb.compute_adfs_and_grads(xy_boundary)
        phi_x_boundary = np.expand_dims(phi_grad_boundary[0, :, 0], axis=1)
        phi_y_boundary = np.expand_dims(phi_grad_boundary[0, :, 1], axis=1)


        term_a_phi = (np.expand_dims(internal_quadrature_data.weights, axis=1).T @ phi_internal_laplacian)

        term_b_phi = np.zeros(phi_y_boundary.shape)

        for i in range(phi_internal_laplacian.shape[1]):
            for e in range(polygon_vertices.shape[1]):
                term_b_phi[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] \
                    = (phi_x_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] *
                       polygon_edge_normals[0, e]
                       + phi_y_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] *
                       polygon_edge_normals[1, e])

        term_b_phi = (np.expand_dims(boundary_quadrature_data.quadrature.weights, axis=1).T @ term_b_phi)

        print("error: ", np.linalg.norm(term_a_phi - term_b_phi))

    def test_laplacian_values_phi_map(self):

        """
        No-regularity of segment and adfs change sing to integrals
        """

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
        reference_quadrature = gedim.quadrature.Quadrature_Gauss1D()
        reference_segment_quadrature_data = reference_quadrature.fill_points_and_weights(20)
        reference_triangle_quadrature_data: gedim.quadrature.QuadratureData \
            = gedim.quadrature.Quadrature_Gauss2D_Triangle.fill_points_and_weights(20)

        polygon_vertices = np.array([[0.90, 0.65, -0.35, -0.90, 0.10],
                              [-0.80, 0.86, 0.60, 0.10, -0.60],
                              [0.0, 0.0, 0.0, 0.0, 0.0]])

        polygon_edge_lengths = geometry_utilities.polygon_edge_lengths(polygon_vertices)
        polygon_edge_tangents = geometry_utilities.polygon_edge_tangents(polygon_vertices)
        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        polygon_area = geometry_utilities.polygon_area(polygon_vertices)
        polygon_diameter = geometry_utilities.polygon_diameter(polygon_vertices)
        polygon_centroid = geometry_utilities.polygon_centroid(polygon_vertices, polygon_area)
        polygon_list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(polygon_vertices, polygon_centroid)
        polygon_triangulations = geometry_utilities.extract_triangulation_points_by_internal_point(polygon_vertices, polygon_centroid, polygon_list_triangles)

        # plot_polygon_with_edge_normals(polygon_vertices, polygon_edge_normals, 0.15)

        boundary_quadrature_data = quadrature.polygon_edges_quadrature(reference_segment_quadrature_data,
                                                              polygon_vertices,
                                                              polygon_edge_lengths,
                                                              [True for _ in range(polygon_vertices.shape[1])],
                                                              polygon_edge_tangents,
                                                              polygon_edge_normals)

        internal_quadrature_data \
            = quadrature.polygon_internal_quadrature(reference_triangle_quadrature_data, polygon_triangulations)


        method_order = 1
        method_type_adf = BoundaryMethodType.line
        bubble_type = BubbleType.product
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_adfs=method_type_adf,
                               bubble_type=bubble_type)

        navem_polygon = NAVEMPolygon(geometry_utilities, polygon_vertices, polygon_vertices, polygon_centroid,
                                     polygon_diameter, polygon_list_triangles)

        vertices = np.array([navem_polygon.mapped_vertices[:2, :]])
        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        eb.initialize_boundary_properties(vertices)

        xy_internal, jac_inertia = navem_polygon.map_inertia_inv(internal_quadrature_data.points)
        xy_internal = xy_internal[:2, :].T
        xy_internal = tf.convert_to_tensor(xy_internal, dtype=tf.float64)
        xy_internal = tf.expand_dims(xy_internal, 0)
        xy_internal = tf.concat([xy_internal] * vertices.shape[0], axis=0)

        xy_boundary, _ = navem_polygon.map_inertia_inv(boundary_quadrature_data.quadrature.points)
        xy_boundary = xy_boundary[:2, :].T
        xy_boundary = tf.convert_to_tensor(xy_boundary, dtype=tf.float64)
        xy_boundary = tf.expand_dims(xy_boundary, 0)
        xy_boundary = tf.concat([xy_boundary] * vertices.shape[0], axis=0)

        jac_per_pol = np.zeros(shape=(1, 2, 2, 1))
        for i in range(1):
            jac_per_pol[0, :, :, i] = jac_inertia[:2, :2]
        jac_per_pol = tf.convert_to_tensor(jac_per_pol, dtype=tf.float64)

        phi_internal, phi_grad_internal, phi_sec_ders_internal = eb.compute_adfs_and_grads_and_second_derivatives(xy_internal)
        phi_sec_ders_internal = eb.map_adf_second_derivatives(phi_sec_ders_internal, jac_per_pol)
        phi_internal_laplacian = np.expand_dims((phi_sec_ders_internal[0, :, 0, 0] + phi_sec_ders_internal[0, :, 0, 2]).numpy(), axis=1)
        phi_boundary, phi_grad_boundary = eb.compute_adfs_and_grads(xy_boundary)
        phi_grad_boundary = eb.map_adf_grads(phi_grad_boundary, jac_per_pol)
        phi_x_boundary = np.expand_dims(phi_grad_boundary[0, :, 0, 0], axis=1)
        phi_y_boundary = np.expand_dims(phi_grad_boundary[0, :, 0, 1], axis=1)


        term_a_phi = (np.expand_dims(internal_quadrature_data.weights, axis=1).T @ phi_internal_laplacian)

        term_b_phi = np.zeros(phi_y_boundary.shape)
        num_points_on_each_edge = reference_segment_quadrature_data.points.shape[1]
        for i in range(phi_internal_laplacian.shape[1]):
            for e in range(polygon_vertices.shape[1]):
                term_b_phi[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] \
                    = (phi_x_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] *
                       polygon_edge_normals[0, e]
                       + phi_y_boundary[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] *
                       polygon_edge_normals[1, e])

        term_b_phi = (np.expand_dims(boundary_quadrature_data.quadrature.weights, axis=1).T @ term_b_phi)

        print("error: ", np.linalg.norm(term_a_phi - term_b_phi))

    def test_lagrange_values_1d(self):

        method_order = 2
        reference_do_fs_coordinates = chebyshev_lobatto_nodes(0, 1, method_order + 1)
        reference_do_fs_coordinates = tf.convert_to_tensor(reference_do_fs_coordinates, dtype=tf.float64)
        lagrange_1d_coefficients = polydim.interpolation.lagrange.lagrange_1_d_coefficients(reference_do_fs_coordinates.numpy())

        evaluation_points_x = tf.convert_to_tensor(np.array([np.linspace(0.0, 1.0, 100)]), dtype=tf.float64)

        values = lagrange_1d_values(reference_do_fs_coordinates, lagrange_1d_coefficients, evaluation_points_x)
        values_do_fs = lagrange_1d_values(reference_do_fs_coordinates, lagrange_1d_coefficients, tf.convert_to_tensor(np.array([reference_do_fs_coordinates]), dtype=tf.float64))

        self.assertTrue(tf.linalg.norm(values_do_fs - np.eye(method_order + 1)) < 1e-10)

        dof_id = 2
        plt.plot(evaluation_points_x[0, :], values[0, :, dof_id], 'b')
        plt.plot(reference_do_fs_coordinates, values_do_fs[0, :, dof_id], 'ro')
        plt.show()

    def test_pentagon_wachspress(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[0.90, 0.65, -0.35, -0.90, 0.10],
                              [-0.80, 0.86, 0.60, 0.10, -0.60]]
                             ])

        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        n = 151
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        x_all = np.expand_dims(np.tile(x, n), 1)
        y_all = np.expand_dims(np.repeat(y, n), 1)
        xy = np.concatenate([x_all, y_all], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        method_order = 4
        method_type_g = BoundaryMethodType.wachspress
        method_type_adf = BoundaryMethodType.segment
        bubble_type = BubbleType.approximate_distance_function
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_g=method_type_g,
                               boundary_method_type_adfs=method_type_adf,
                               bubble_type=bubble_type)

        num_boundary_do_fs = method_order * vertices.shape[1]

        eb.initialize_boundary_properties(vertices)
        g = eb.compute_g(xy)
        phi = eb.compute_adfs(xy)

        for pol_id in range(vertices.shape[0]):
            eb.draw_adf(xy, phi, pol_id, scatter_plot=False, contour_plot=True)

            for dof_id in range(num_boundary_do_fs):
                eb.draw_g(xy, g, pol_id, dof_id, scatter_plot=False, contour_plot=True)

    def test_pentagon_segment(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[0.90, 0.65, -0.35, -0.90, 0.10],
                              [-0.80, 0.86, 0.60, 0.10, -0.60]],

                             [[0.80, 0.55, -0.15, -0.70, -0.20],
                              [-0.60, 0.8, 0.90, 0.30, -0.05]],

                             [[0.80, 0.55, 0.15, -0.70, -0.20],
                              [-0.60, 0.8, 0.30, 0.30, -0.50]],

                             [[0.50, -0.5, -0.55, -0.50, -0.30],
                              [-0.50, 0.5, 0.45, 0.10, -0.70]],

                             [[0.5, 0.5, 0.5, -0.90, -0.2],
                              [-0.8, 0.0, 0.8, 0.0, -0.4]]
                             ])

        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        n = 151
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        x_all = np.expand_dims(np.tile(x, n), 1)
        y_all = np.expand_dims(np.repeat(y, n), 1)
        xy = np.concatenate([x_all, y_all], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        method_type = BoundaryMethodType.segment
        bubble_type = BubbleType.approximate_distance_function

        method_order = 1
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_adfs=method_type,
                               boundary_method_type_g=method_type,
                               bubble_type=bubble_type)

        eb.initialize_boundary_properties(vertices)
        g = eb.compute_g(xy)
        phi = eb.compute_adfs(xy)

        num_boundary_do_fs = method_order * vertices.shape[1]
        for pol_id in range(vertices.shape[0]):
           eb.draw_adf(xy, phi, pol_id, scatter_plot=False, contour_plot=True)

           for dof_id in range(num_boundary_do_fs):
                eb.draw_g(xy, g, pol_id, dof_id, scatter_plot=False, contour_plot=True)
                eb.draw_g_on_one_edge(xy, pol_id, dof_id)

    def test_hexagon_segment_product(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[0.5, 0.5, 0.5, -0.90, 0.5, 0.5],
                              [0.0, 0.4, 0.8, 0.0, -0.8, -0.4]]
                             ])


        method_type = BoundaryMethodType.segment
        bubble_type = BubbleType.product

        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        n = 151
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        x_all = np.expand_dims(np.tile(x, n), 1)
        y_all = np.expand_dims(np.repeat(y, n), 1)
        xy = np.concatenate([x_all, y_all], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        method_order = 1
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_adfs=method_type,
                               boundary_method_type_g=method_type,
                               bubble_type=bubble_type)

        eb.initialize_boundary_properties(vertices)
        g = eb.compute_g(xy)
        phi = eb.compute_adfs(xy)

        num_boundary_do_fs = method_order * vertices.shape[1]
        for pol_id in range(vertices.shape[0]):
            eb.draw_adf(xy, phi, pol_id, scatter_plot=False, contour_plot=True)

            for dof_id in range(num_boundary_do_fs):
                eb.draw_g(xy, g, pol_id, dof_id, scatter_plot=False, contour_plot=True)
                eb.draw_g_on_one_edge(xy, pol_id, dof_id)

    def test_hexagon_segment_adf(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[0.5, 0.5, 0.5, -0.90, 0.5, 0.5],
                              [0.0, 0.4, 0.8, 0.0, -0.8, -0.4]]
                             ])

        method_type = BoundaryMethodType.segment
        bubble_type = BubbleType.approximate_distance_function

        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        n = 151
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        x_all = np.expand_dims(np.tile(x, n), 1)
        y_all = np.expand_dims(np.repeat(y, n), 1)
        xy = np.concatenate([x_all, y_all], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        method_order = 1
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_adfs=method_type,
                               boundary_method_type_g=method_type,
                               bubble_type=bubble_type)

        eb.initialize_boundary_properties(vertices)
        g = eb.compute_g(xy)
        phi = eb.compute_adfs(xy)

        num_boundary_do_fs = method_order * vertices.shape[1]
        for pol_id in range(vertices.shape[0]):
            eb.draw_adf(xy, phi, pol_id, scatter_plot=False, contour_plot=True)

            for dof_id in range(num_boundary_do_fs):
                eb.draw_g(xy, g, pol_id, dof_id, scatter_plot=False, contour_plot=True)
                eb.draw_g_on_one_edge(xy, pol_id, dof_id)

    def test_quad_line(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[-0.5, 0.5, 0.5, -0.5],
                              [0.0, 0.0, 1.0, 1.0]],

                             [[-0.5, 0.0, 0.5, -0.5],
                              [-0.5, -0.5, -0.5, 0.5]]
                             ])

        method_type = BoundaryMethodType.line
        bubble_type = BubbleType.approximate_distance_function

        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        n = 151
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        x_all = np.expand_dims(np.tile(x, n), 1)
        y_all = np.expand_dims(np.repeat(y, n), 1)
        xy = np.concatenate([x_all, y_all], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        method_order = 1
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_adfs=method_type,
                               boundary_method_type_g=method_type,
                               bubble_type=bubble_type)

        eb.initialize_boundary_properties(vertices)
        g = eb.compute_g(xy)
        phi = eb.compute_adfs(xy)

        num_boundary_do_fs = method_order * vertices.shape[1]
        for pol_id in range(vertices.shape[0]):
            eb.draw_adf(xy, phi, pol_id, scatter_plot=False, contour_plot=True)

            for dof_id in range(num_boundary_do_fs):
                eb.draw_g(xy, g, pol_id, dof_id, scatter_plot=False, contour_plot=True)
                eb.draw_g_on_one_edge(xy, pol_id, dof_id)

    def test_quad_segment(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[-0.5, 0.5, 0.5, -0.5],
                              [0.0, 0.0, 1.0, 1.0]],

                             [[-0.5, 0.0, 0.5, -0.5],
                              [-0.5, -0.5, -0.5, 0.5]]
                             ])

        method_type = BoundaryMethodType.segment
        bubble_type = BubbleType.product

        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        n = 151
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        x_all = np.expand_dims(np.tile(x, n), 1)
        y_all = np.expand_dims(np.repeat(y, n), 1)
        xy = np.concatenate([x_all, y_all], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        method_order = 1
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_adfs=method_type,
                               boundary_method_type_g=method_type,
                               bubble_type=bubble_type)

        eb.initialize_boundary_properties(vertices)
        g = eb.compute_g(xy)
        phi = eb.compute_adfs(xy)

        num_boundary_do_fs = method_order * vertices.shape[1]
        for pol_id in range(vertices.shape[0]):
            eb.draw_adf(xy, phi, pol_id, scatter_plot=False, contour_plot=True)

            for dof_id in range(num_boundary_do_fs):
                eb.draw_g(xy, g, pol_id, dof_id, scatter_plot=False, contour_plot=True)
                eb.draw_g_on_one_edge(xy, pol_id, dof_id)

    def test_pentagon_high_order(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[0.90, 0.65, -0.35, -0.90, 0.10],
                              [-0.80, 0.86, 0.60, 0.10, -0.60]],
                             ])

        method_type = BoundaryMethodType.segment
        bubble_type = BubbleType.approximate_distance_function

        vertices = tf.convert_to_tensor(np.transpose(vertices, axes=[0, 2, 1]), dtype=tf.float64)

        n = 151
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        x_all = np.expand_dims(np.tile(x, n), 1)
        y_all = np.expand_dims(np.repeat(y, n), 1)
        xy = np.concatenate([x_all, y_all], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        method_order = 1
        eb = EnforcingBoundary(geometry_utilities,
                               method_order=method_order,
                               boundary_method_type_adfs=method_type,
                               boundary_method_type_g=method_type,
                               bubble_type=bubble_type)

        eb.initialize_boundary_properties(vertices)
        g = eb.compute_g(xy)
        phi = eb.compute_adfs(xy)

        num_boundary_do_fs = method_order * vertices.shape[1]
        for pol_id in range(vertices.shape[0]):
            eb.draw_adf(xy, phi, pol_id, scatter_plot=True, contour_plot=True)

            for dof_id in range(num_boundary_do_fs):
                eb.draw_g(xy, g, pol_id, dof_id, scatter_plot=True, contour_plot=True)
                eb.draw_g_on_one_edge(xy, pol_id, dof_id)


if __name__ == '__main__':
    unittest.main()
