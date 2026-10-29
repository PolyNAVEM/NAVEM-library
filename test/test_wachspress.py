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

import tensorflow as tf

from NAVEM.PCC_2D.NAVEM_Data_PCC_2D import NAVEMType, BasisFunctionType
from NAVEM.Utilities.points_generator import *
from pypolydim import gedim
from NAVEM.Utilities.wachspress_coordinate import wachspress_weights, compute_g_wachspress
from NAVEM.NeuralNetwork.w_navem_network import WNAVEMNetwork
from NAVEM.Utilities.enforcing_boundary_functions import BubbleType, BoundaryMethodType
from NAVEM.PCC_2D.NAVEM_Data_PCC_2D import NAVEMElementType, BasisFunctionType, NAVEMMappingType
from NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import set_flags, Flags


try:
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True


def boundary_function_1(_edge_idx, x, y):
    """
    Define the boundary function for any given edge
    """
    return np.cos(x) + y**2 + x**2

def boundary_function_2(_edge_idx, x, y):
    """
    Define the boundary function for any given edge
    """
    return np.sin(x) * np.exp(3) - 10 + 5 * y

class TestEnforcingBoundaryFunctionsEfficient(unittest.TestCase):

    def test_wachspress_partition_of_unity(self):
        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-12
        geometry_utilities_config.tolerance2_d = 1.0e-14
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([
            [[0.90, 0.65, -0.35, -0.90, 0.10],
             [-0.80, 0.86, 0.60, 0.10, -0.60],
             [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 1.5, 0.5, -0.5],
             [0.0, 0.0, 1.0, 2.0, 1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.80, 0.55, -0.15, -0.70, -0.20],
             [-0.60, 0.8, 0.90, 0.30, -0.05],
             [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.50, -0.5, -0.55, -0.50, -0.30],
             [-0.50, 0.5, 0.45, 0.10, -0.70],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
        ])

        internal_point = np.array([
            [[0.25], [0.0], [0.0]],
            [[0.5], [1.0], [0.0]],
            [[0.2], [0.3], [0.0]],
            [[-0.2], [-0.2], [0.0]],
        ])

        num_vertices = vertices.shape[2]

        uniform_boundary = False
        uniform_scale = True
        accumulating_power = 0.75
        accumulating_border = BorderType.no_borders
        reference_nodes = grid_over_triangle(PointsTriangleDistributionType.accumulated,
                                             5,
                                             uniform_boundary=uniform_boundary,
                                             uniform_rescale=uniform_scale,
                                             accumulating_power=accumulating_power,
                                             accumulating_border=accumulating_border,
                                             polygon=num_vertices > 3)

        num_points_per_polygon = reference_nodes.shape[1] * num_vertices
        points = np.zeros([vertices.shape[0], num_points_per_polygon, 2])
        for c in range(vertices.shape[0]):
            list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertices[c, :, :],
                                                                                        internal_point[c, :, :])
            triangles_points = geometry_utilities.extract_triangulation_points_by_internal_point(vertices[c, :, :],
                                                                                                 internal_point[
                                                                                                     c, :, :],
                                                                                                 list_triangles)

            # Loss will be minimized over the inertia polygon
            points[c, :, :] = grid_over_polygon(PointsTriangleDistributionType.uniform,
                                                reference_nodes,
                                                triangles_points,
                                                uniform_boundary,
                                                accumulating_border)[:2, :].T

        vertices = tf.convert_to_tensor(np.transpose(vertices[:, :2, :], axes=[0, 2, 1]), dtype=tf.float64)
        points = tf.convert_to_tensor(points, dtype=tf.float64)

        lambdas = wachspress_weights(points, vertices)

        for c in range(vertices.shape[0]):

            self.assertTrue(np.linalg.norm(tf.reduce_sum(lambdas[c, :, :], axis=1).numpy() - np.ones(lambdas.shape[1])) < 1.0e-12)
    #

    def test_wachspress_boundary_function_1(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-12
        geometry_utilities_config.tolerance2_d = 1.0e-14
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([
            [[0.90, 0.65, -0.35, -0.90, 0.10],
             [-0.80, 0.86, 0.60, 0.10, -0.60],
             [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 1.5, 0.5, -0.5],
             [0.0, 0.0, 1.0, 2.0, 1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],

            # [[0.80, 0.55, -0.15, -0.70, -0.20],
            #  [-0.60, 0.8, 0.90, 0.30, -0.05],
            #  [0.0, 0.0, 0.0, 0.0, 0.0]],
            #
            # [[0.50, -0.5, -0.55, -0.50, -0.30],
            #  [-0.50, 0.5, 0.45, 0.10, -0.70],
            #  [0.0, 0.0, 0.0, 0.0, 0.0]],
        ])

        internal_point = np.array([
            [[0.25], [0.0], [0.0]],
            [[0.5], [1.0], [0.0]],
            # [[0.2], [0.3], [0.0]],
            # [[-0.2], [-0.2], [0.0]],
        ])

        num_vertices = vertices.shape[2]

        uniform_boundary = False
        uniform_scale = True
        accumulating_power = 0.75
        accumulating_border = BorderType.all_borders
        reference_nodes = grid_over_triangle(PointsTriangleDistributionType.accumulated,
                                             2,
                                             uniform_boundary=uniform_boundary,
                                             uniform_rescale=uniform_scale,
                                             accumulating_power=accumulating_power,
                                             accumulating_border=accumulating_border,
                                             polygon=num_vertices > 3)

        num_points_per_polygon = reference_nodes.shape[1] * num_vertices
        points = np.zeros([vertices.shape[0], num_points_per_polygon, 2])
        for c in range(vertices.shape[0]):
            list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertices[c, :, :],
                                                                                        internal_point[c, :, :])
            triangles_points = geometry_utilities.extract_triangulation_points_by_internal_point(vertices[c, :, :],
                                                                                                 internal_point[
                                                                                                     c, :, :],
                                                                                                 list_triangles)

            # Loss will be minimized over the inertia polygon
            points[c, :, :] = grid_over_polygon(PointsTriangleDistributionType.uniform,
                                                reference_nodes,
                                                triangles_points,
                                                uniform_boundary,
                                                accumulating_border)[:2, :].T

        vertices = tf.convert_to_tensor(np.transpose(vertices[:, :2, :], axes=[0, 2, 1]), dtype=tf.float64)
        points = tf.convert_to_tensor(points, dtype=tf.float64)

        lambdas = wachspress_weights(points, vertices)
        lifting_g = compute_g_wachspress(vertices, lambdas, boundary_function_1)


        for c in range(vertices.shape[0]):

            f1 = boundary_function_1(None, points[c, :, 0], points[c, :, 1])
            diff_rete_g = lifting_g[c, :] - f1

            print(diff_rete_g)

            # plot 3D surface
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')

            sc = ax.scatter(
                points[c, :, 0],
                points[c, :, 1],
                diff_rete_g,
                c=diff_rete_g,
                cmap='viridis',
                s=40
            )
            plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("diff_rete_g")

            plt.show()

    def test_wachspress_boundary_function_2(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-12
        geometry_utilities_config.tolerance2_d = 1.0e-14
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([
            [[0.90, 0.65, -0.35, -0.90, 0.10],
             [-0.80, 0.86, 0.60, 0.10, -0.60],
             [0.0, 0.0, 0.0, 0.0, 0.0]],

            # [[0.0, 1.0, 1.5, 0.5, -0.5],
            #  [0.0, 0.0, 1.0, 2.0, 1.0],
            #  [0.0, 0.0, 0.0, 0.0, 0.0]],
            #
            # [[0.80, 0.55, -0.15, -0.70, -0.20],
            #  [-0.60, 0.8, 0.90, 0.30, -0.05],
            #  [0.0, 0.0, 0.0, 0.0, 0.0]],
            #
            # [[0.50, -0.5, -0.55, -0.50, -0.30],
            #  [-0.50, 0.5, 0.45, 0.10, -0.70],
            #  [0.0, 0.0, 0.0, 0.0, 0.0]],
        ])

        internal_point = np.array([
            [[0.25], [0.0], [0.0]],
            # [[0.5], [1.0], [0.0]],
            # [[0.2], [0.3], [0.0]],
            # [[-0.2], [-0.2], [0.0]],
        ])

        num_vertices = vertices.shape[2]

        uniform_boundary = False
        uniform_scale = True
        accumulating_power = 0.75
        accumulating_border = BorderType.all_borders
        reference_nodes = grid_over_triangle(PointsTriangleDistributionType.accumulated,
                                             5,
                                             uniform_boundary=uniform_boundary,
                                             uniform_rescale=uniform_scale,
                                             accumulating_power=accumulating_power,
                                             accumulating_border=accumulating_border,
                                             polygon=num_vertices > 3)

        num_points_per_polygon = reference_nodes.shape[1] * num_vertices
        points = np.zeros([vertices.shape[0], num_points_per_polygon, 2])
        for c in range(vertices.shape[0]):
            list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertices[c, :, :],
                                                                                        internal_point[c, :, :])
            triangles_points = geometry_utilities.extract_triangulation_points_by_internal_point(vertices[c, :, :],
                                                                                                 internal_point[
                                                                                                     c, :, :],
                                                                                                 list_triangles)

            # Loss will be minimized over the inertia polygon
            points[c, :, :] = grid_over_polygon(PointsTriangleDistributionType.uniform,
                                                reference_nodes,
                                                triangles_points,
                                                uniform_boundary,
                                                accumulating_border)[:2, :].T

        # num_points_per_polygon = 1
        # points = np.zeros([vertices.shape[0], num_points_per_polygon, 2])
        # for c in range(vertices.shape[0]):
        #     points[c, :, :] = vertices[c, 0:2, 0].T


        vertices = tf.convert_to_tensor(np.transpose(vertices[:, :2, :], axes=[0, 2, 1]), dtype=tf.float64)
        points = tf.convert_to_tensor(points, dtype=tf.float64)

        lambdas = wachspress_weights(points, vertices)
        lifting_g = compute_g_wachspress(vertices, lambdas, boundary_function_2)

        for c in range(vertices.shape[0]):

            f1 = boundary_function_2(None, points[c, :, 0], points[c, :, 1])
            diff_rete_g = lifting_g[c, :] - f1

            print(diff_rete_g)

            # plot 3D surface
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')

            sc = ax.scatter(
                points[c, :, 0],
                points[c, :, 1],
                diff_rete_g,
                c=diff_rete_g,
                cmap='viridis',
                s=40
            )
            plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("g")


            plt.show()

    def test_lifting_network(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-12
        geometry_utilities_config.tolerance2_d = 1.0e-14
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([
            [[0.90, 0.65, -0.35, -0.90, 0.10],
             [-0.80, 0.86, 0.60, 0.10, -0.60],
             [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 1.5, 0.5, -0.5],
             [0.0, 0.0, 1.0, 2.0, 1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],

            # [[0.80, 0.55, -0.15, -0.70, -0.20],
            #  [-0.60, 0.8, 0.90, 0.30, -0.05],
            #  [0.0, 0.0, 0.0, 0.0, 0.0]],
            #
            # [[0.50, -0.5, -0.55, -0.50, -0.30],
            #  [-0.50, 0.5, 0.45, 0.10, -0.70],
            #  [0.0, 0.0, 0.0, 0.0, 0.0]],
        ])

        internal_point = np.array([
            [[0.25], [0.0], [0.0]],
            [[0.5], [1.0], [0.0]],
            # [[0.2], [0.3], [0.0]],
            # [[-0.2], [-0.2], [0.0]],
        ])

        num_vertices = vertices.shape[2]

        uniform_boundary = False
        uniform_scale = True
        accumulating_power = 0.75
        accumulating_border = BorderType.all_borders
        distribution_points_type = PointsTriangleDistributionType.accumulated
        reference_nodes = grid_over_triangle(distribution_points_type,
                                             2,
                                             uniform_boundary=uniform_boundary,
                                             uniform_rescale=uniform_scale,
                                             accumulating_power=accumulating_power,
                                             accumulating_border=accumulating_border,
                                             polygon=num_vertices > 3)

        num_points_per_polygon = reference_nodes.shape[1] * num_vertices
        num_training_polygons = vertices.shape[0]
        num_functions_per_polygon = 1
        network_input_dimension = 2 + 2 * 5
        inputs = np.zeros((num_training_polygons * num_points_per_polygon * num_functions_per_polygon,
                           network_input_dimension))
        for c in range(vertices.shape[0]):
            for v_id in range(num_functions_per_polygon):
                # rotated_vertices = np.roll(vertices[c, :, :], axis=1, shift=-v_id)
                rotated_vertices = vertices[c, :, :]
                list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(rotated_vertices,
                                                                                            internal_point[c, :, :])
                triangles_points = geometry_utilities.extract_triangulation_points_by_internal_point(rotated_vertices,
                                                                                                     internal_point[
                                                                                                         c, :, :],
                                                                                                     list_triangles)

                # Loss will be minimized over the inertia polygon
                points = grid_over_polygon(PointsTriangleDistributionType.uniform,
                                           reference_nodes,
                                           triangles_points,
                                           uniform_boundary,
                                           accumulating_border)[:2, :].T

                flat_rotated_vertices = rotated_vertices[:2, :].flatten()
                rep_flat_rot_vertices = np.tile(flat_rotated_vertices[np.newaxis, :], [num_points_per_polygon, 1])
                c_inputs = np.concatenate([points, rep_flat_rot_vertices], axis=1)

                inputs[(c * num_functions_per_polygon + v_id) * num_points_per_polygon:
                       (c * num_functions_per_polygon + v_id + 1) * num_points_per_polygon, :] = c_inputs


        inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)

        flags: Flags = set_flags(network_input_dimension,
                                 1,
                                 NAVEMType.W_NAVEM,
                                 BasisFunctionType.vertex,
                                 num_functions_per_polygon,
                                 NAVEMElementType.generic_convex,
                                 num_vertices,
                                 './',
                                 num_training_polygons,
                                 2,
                                 5,
                                 50,
                                 50,
                                 1.0e-03,
                                 1.0e-03,
                                 False,
                                 1.0e-08,
                                 './',
                                 False,
                                 10,
                                 distribution_points_type.value,
                                 BoundaryMethodType.wachspress.value,
                                 BoundaryMethodType.segment.value,
                                 BubbleType.approximate_distance_function.value)


        nn = WNAVEMNetwork(flags, in_training=True)

        lifting_g = nn.compute_network_lifting(inputs, boundary_function_1).numpy()
        points = inputs[:, :2].numpy()

        lifting_g = np.reshape(lifting_g, (num_training_polygons, num_points_per_polygon))
        points = np.reshape(points, (num_training_polygons, num_points_per_polygon, 2))

        for c in range(vertices.shape[0]):

            f1 = boundary_function_1(None, points[c, :, 0], points[c, :, 1])
            diff_rete_g = lifting_g[c, :] - f1

            print(diff_rete_g)

            # plot 3D surface
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')

            sc = ax.scatter(
                points[c, :, 0],
                points[c, :, 1],
                diff_rete_g,
                c=diff_rete_g,
                cmap='viridis',
                s=40
            )
            plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("g")


            plt.show()


if __name__ == '__main__':
    unittest.main()
