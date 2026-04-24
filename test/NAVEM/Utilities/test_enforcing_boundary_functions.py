import unittest
from src.NAVEM.Utilities.enforcing_boundary_functions import EnforcingBoundary
import tensorflow as tf
import numpy as np
from pypolydim import gedim
import os

try:
    import matplotlib.pyplot as plt
    from matplotlib import use
    import importlib
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True


class TestEnforcingBoundaryFunctionsEfficient(unittest.TestCase):

    def test_pentagon_line(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[ 0.90, 0.65, -0.35, -0.90,  0.10],
                              [-0.80, 0.86,  0.60,  0.10, -0.60]],

                              [[0.80, 0.55, -0.15, -0.70, -0.20],
                              [-0.60, 0.8, 0.90, 0.30, -0.05]],

                              [[0.80, 0.55, 0.15, -0.70, -0.20],
                              [-0.60, 0.8, 0.30, 0.30, -0.50]],

                              [[0.50, -0.5, -0.55, -0.50, -0.30],
                              [-0.50, 0.5, 0.45, 0.10, -0.70]],

                              [[0.5, 0.5, 0.5, -0.90,  -0.2],
                              [-0.8, 0.0, 0.8,  0.0, -0.4]]
                              ])

        N = 151
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        xall = np.expand_dims(np.tile(x, N), 1)
        yall = np.expand_dims(np.repeat(y, N), 1)
        xy = np.concatenate([xall, yall], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        n_points_per_pol = xy.shape[1]
        jac_per_pol = np.zeros(shape=(vertices.shape[0], 2, 2, vertices.shape[2]))

        method_type = 0
        for p in range(vertices.shape[0]):
            for v in range(vertices.shape[2]):
                jac_per_pol[p, :, :, v] = np.eye(2)

        draw_lifting = False
        eb = EnforcingBoundary(geometry_utilities)
        eb.prepare_using_vertices(vertices, jac_per_pol)

        export_file_path = './Export/test_pentagon_line'
        if not os.path.exists(export_file_path):
            os.makedirs(export_file_path)

        for pol_id in range(vertices.shape[0]):
            dof_id = 0
            eb.draw_function(N, x, y, xy, pol_id, dof_id, draw_lifting, export_file_path, method_type=method_type, bubble_type = 2)
            #eb.draw_sec_der_function(N, x, y, xy, pol_id, dof_id, draw_lifting)
            #eb.scatter_border(pol_id, dof_id, method_type, plot_all_edges=True, plot_inner_pts=True, draw_lifting=draw_lifting)
            eb.draw_function_one_edge(N, x, y, xy, pol_id=pol_id, edge_id=dof_id, method_type=method_type, file_path = export_file_path)

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

        N = 151
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        xall = np.expand_dims(np.tile(x, N), 1)
        yall = np.expand_dims(np.repeat(y, N), 1)
        xy = np.concatenate([xall, yall], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        n_points_per_pol = xy.shape[1]
        jac_per_pol = np.zeros(shape=(vertices.shape[0], 2, 2, vertices.shape[2]))

        method_type = 1
        for p in range(vertices.shape[0]):
            for v in range(vertices.shape[2]):
                jac_per_pol[p, :, :, v] = np.eye(2)

        draw_lifting = False
        eb = EnforcingBoundary(geometry_utilities)
        eb.prepare_using_vertices(vertices, jac_per_pol)

        export_file_path = './Export/test_pentagon_segment'
        if not os.path.exists(export_file_path):
            os.makedirs(export_file_path)

        for pol_id in range(vertices.shape[0]):
            dof_id = 0
            eb.draw_function(N, x, y, xy, pol_id, dof_id, draw_lifting, method_type=method_type, bubble_type=1, file_path = export_file_path)
            # eb.draw_sec_der_function(N, x, y, xy, pol_id, dof_id, draw_lifting)
            # eb.scatter_border(pol_id, dof_id, method_type, plot_all_edges=True, plot_inner_pts=True, draw_lifting=draw_lifting)
            eb.draw_function_one_edge(N, x, y, xy, pol_id=pol_id, edge_id=dof_id, method_type=method_type, file_path = export_file_path)


    def test_hexagon_line(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[ 0.5, 0.5, 0.5, -0.90,  0.5, 0.5],
                              [ 0.0, 0.4, 0.8,  0.0, -0.8, -0.4]]
                              ])

        N = 151
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        xall = np.expand_dims(np.tile(x, N), 1)
        yall = np.expand_dims(np.repeat(y, N), 1)
        xy = np.concatenate([xall, yall], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        n_points_per_pol = xy.shape[1]
        jac_per_pol = np.zeros(shape=(vertices.shape[0], 2, 2, vertices.shape[2]))

        method_type = 0
        for p in range(vertices.shape[0]):
            for v in range(vertices.shape[2]):
                jac_per_pol[p, :, :, v] = np.eye(2)

        draw_lifting = False
        eb = EnforcingBoundary(geometry_utilities)
        eb.prepare_using_vertices(vertices, jac_per_pol)

        export_file_path = './Export/test_hexagon_line'
        if not os.path.exists(export_file_path):
            os.makedirs(export_file_path)

        for pol_id in range(vertices.shape[0]):
            dof_id = 0
            eb.draw_function(N, x, y, xy, pol_id, dof_id, draw_lifting, method_type=method_type, bubble_type=2, file_path = export_file_path)
            # eb.draw_sec_der_function(N, x, y, xy, pol_id, dof_id, draw_lifting)
            # eb.scatter_border(pol_id, dof_id, method_type, plot_all_edges=True, plot_inner_pts=True, draw_lifting=draw_lifting)
            eb.draw_function_one_edge(N, x, y, xy, pol_id=pol_id, edge_id=dof_id, method_type=method_type, file_path = export_file_path)


    def test_hexagon_segment(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[ 0.5, 0.5, 0.5, -0.90,  0.5, 0.5],
                              [ 0.0, 0.4, 0.8,  0.0, -0.8, -0.4]]
                              ])

        N = 151
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        xall = np.expand_dims(np.tile(x, N), 1)
        yall = np.expand_dims(np.repeat(y, N), 1)
        xy = np.concatenate([xall, yall], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        n_points_per_pol = xy.shape[1]
        jac_per_pol = np.zeros(shape=(vertices.shape[0], 2, 2, vertices.shape[2]))

        method_type = 1
        for p in range(vertices.shape[0]):
            for v in range(vertices.shape[2]):
                jac_per_pol[p, :, :, v] = np.eye(2)

        draw_lifting = False
        eb = EnforcingBoundary(geometry_utilities)
        eb.prepare_using_vertices(vertices, jac_per_pol)

        export_file_path = './Export/test_hexagon_segment'
        if not os.path.exists(export_file_path):
            os.makedirs(export_file_path)

        for pol_id in range(vertices.shape[0]):
            dof_id = 5
            eb.draw_function(N, x, y, xy, pol_id, dof_id, draw_lifting, method_type=method_type, bubble_type=1, file_path = export_file_path)
            # eb.draw_sec_der_function(N, x, y, xy, pol_id, dof_id, draw_lifting)
            # eb.scatter_border(pol_id, dof_id, method_type, plot_all_edges=True, plot_inner_pts=True, draw_lifting=draw_lifting)
            eb.draw_function_one_edge(N, x, y, xy, pol_id=pol_id, edge_id=dof_id, method_type=method_type, file_path = export_file_path)


    def test_quad_line(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[-0.5, 0.5, 0.5, -0.5],
                              [0.0, 0.0, 1.0, 1.0]],

                             [[-0.5, 0.0, 0.5, -0.5],
                              [-0.5, -0.5, -0.5, 0.5]]
                             ])

        N = 151
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        xall = np.expand_dims(np.tile(x, N), 1)
        yall = np.expand_dims(np.repeat(y, N), 1)
        xy = np.concatenate([xall, yall], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        n_points_per_pol = xy.shape[1]
        jac_per_pol = np.zeros(shape=(vertices.shape[0], 2, 2, vertices.shape[2]))

        method_type = 0
        for p in range(vertices.shape[0]):
            for v in range(vertices.shape[2]):
                jac_per_pol[p, :, :, v] = np.eye(2)

        draw_lifting = False
        eb = EnforcingBoundary(geometry_utilities)
        eb.prepare_using_vertices(vertices, jac_per_pol)

        export_file_path = './Export/test_quad_line'
        if not os.path.exists(export_file_path):
            os.makedirs(export_file_path)

        for pol_id in range(vertices.shape[0]):
            dof_id = 0
            eb.draw_function(N, x, y, xy, pol_id, dof_id, draw_lifting, method_type=method_type, bubble_type=2, file_path = export_file_path)
            # eb.draw_sec_der_function(N, x, y, xy, pol_id, dof_id, draw_lifting)
            # eb.scatter_border(pol_id, dof_id, method_type, plot_all_edges=True, plot_inner_pts=True, draw_lifting=draw_lifting)
            eb.draw_function_one_edge(N, x, y, xy, pol_id=pol_id, edge_id=dof_id, method_type=method_type, file_path = export_file_path)

    def test_quad_segment(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        vertices = np.array([[[-0.5, 0.5, 0.5, -0.5],
                              [0.0, 0.0, 1.0, 1.0]],

                             [[-0.5, 0.0, 0.5, -0.5],
                              [-0.5, -0.5, -0.5, 0.5]]
                             ])

        N = 151
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        xall = np.expand_dims(np.tile(x, N), 1)
        yall = np.expand_dims(np.repeat(y, N), 1)
        xy = np.concatenate([xall, yall], axis=1)
        xy = tf.convert_to_tensor(xy, dtype=tf.float64)

        xy = tf.expand_dims(xy, 0)
        xy = tf.concat([xy] * vertices.shape[0], axis=0)

        n_points_per_pol = xy.shape[1]
        jac_per_pol = np.zeros(shape=(vertices.shape[0], 2, 2, vertices.shape[2]))

        method_type = 1
        for p in range(vertices.shape[0]):
            for v in range(vertices.shape[2]):
                jac_per_pol[p, :, :, v] = np.eye(2)

        draw_lifting = False
        eb = EnforcingBoundary(geometry_utilities)
        eb.prepare_using_vertices(vertices, jac_per_pol)

        export_file_path = './Export/test_quad_segment'
        if not os.path.exists(export_file_path):
            os.makedirs(export_file_path)

        for pol_id in range(vertices.shape[0]):
            dof_id = 0
            eb.draw_function(N, x, y, xy, pol_id, dof_id, draw_lifting, method_type=method_type, bubble_type=1, file_path = export_file_path)
            # eb.draw_sec_der_function(N, x, y, xy, pol_id, dof_id, draw_lifting)
            # eb.scatter_border(pol_id, dof_id, method_type, plot_all_edges=True, plot_inner_pts=True, draw_lifting=draw_lifting)
            eb.draw_function_one_edge(N, x, y, xy, pol_id=pol_id, edge_id=dof_id, method_type=method_type, file_path = export_file_path)
            eb.draw_trimming_function_one_edge(N, x, y, xy, pol_id=pol_id, edge_id=dof_id, file_path = export_file_path)



if __name__ == '__main__':
    unittest.main()
