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

try:
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True


class TestEnforcingBoundaryFunctionsEfficient(unittest.TestCase):

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
