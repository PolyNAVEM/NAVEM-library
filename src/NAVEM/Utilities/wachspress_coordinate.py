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

import tensorflow as tf
from NAVEM.Utilities.points_generator import *
from pypolydim import gedim

def wachspress_weights(points: tf.Tensor, vertices: tf.Tensor) -> tf.Tensor:
    """
    Compute Wachspress weights as:

        w_i(x) =
            det(n_{i-1}, n_i)
            PROD_{j != i-1,i} h_j(x)

    Parameters
    ----------
    points : (num_points, 2)
    vertices : (num_vertices, 2)

    Returns
    -------
    lambda_w : (num_points, num_vertices)
        normalized Wachspress coordinates
    """


    vertices = tf.convert_to_tensor(vertices, tf.float64)
    points   = tf.convert_to_tensor(points, tf.float64)

    num_polygons = vertices.shape[0]
    num_vertices = vertices.shape[1]
    num_points = points.shape[1]

    v_next = tf.roll(vertices, shift=-1, axis=1)

    # edge vectors
    edge_tangents = v_next - vertices

    # NON normalized normals
    edge_normals = tf.stack([-edge_tangents[:, :, 1], edge_tangents[:, :,0]], axis=2)

    # h_i(x)
    h = tf.reduce_sum(
        (points[:, :, None, :] - vertices[:, None, :, :]) *
        edge_normals[:, None, :, :],
        axis=-1
    )

    prev_edge_normals = tf.roll(edge_normals, shift=1, axis=1)
    det_prev_current = tf.einsum('pij,pij->pi',
                                 prev_edge_normals[:, :, ::-1],
                                 edge_normals * tf.constant([1.0, -1.0], dtype=tf.float64))

    prod_dist = tf.zeros(shape=(num_polygons, num_points, 0), dtype=tf.float64)
    for i in range(num_vertices):

        mask = [j for j in range(num_vertices) if j != i and j != (i-1) % num_vertices]

        prod_im1_i = tf.expand_dims(tf.reduce_prod(tf.gather(h, mask, axis=2), axis=2), axis=2)

        prod_dist = tf.concat([prod_dist, prod_im1_i], axis=2)

    w = prod_dist * det_prev_current[:, None, :]

    return w / tf.reduce_sum(w, axis=2, keepdims=True)


def boundary_function(edge_idx, x, _y):
    """
    Define the boundary function for any given edge
    """
    return tf.where(
                    edge_idx == 2,
                    tf.sin(tf.constant(np.pi, dtype=x.dtype) * x),
                    tf.zeros_like(x)
                )

def compute_g_wachspress(vertices: tf.Tensor, lambdas: tf.Tensor) -> tf.Tensor:
    """
    Compute transfinite interpolant of Dirichlet boundary functions
    """
    num_vertices = vertices.shape[1]
    g = tf.zeros([lambdas.shape[0], lambdas.shape[1]], dtype=tf.float64)

    for i in range(num_vertices):
        """
        get the other vertices for edges attached to vertex i
        """
        ip1 = (i + 1) % num_vertices  # next vertex
        im1 = (i - 1) % num_vertices  # previous vertex

        """
        (x,y) coordinates where the boundary function will be evaluated
        """
        x_ip1 = ((1.0 - lambdas[:, :, ip1]) * vertices[:, None, i, 0]
                 + lambdas[:, :, ip1] * vertices[:, None, ip1, 0])
        y_ip1 = ((1.0 - lambdas[:, :, ip1]) * vertices[:, None, i, 1]
                 + lambdas[:, :, ip1] * vertices[:, None, ip1, 1])
        x_im1 = ((1.0 - lambdas[:, :, im1]) * vertices[:, None, i, 0]
                 + lambdas[:, :, im1] * vertices[:, None, im1, 0])
        y_im1 = ((1.0 - lambdas[:, :, im1]) * vertices[:, None, i, 1]
                 + lambdas[:, :, im1] * vertices[:, None, im1, 1])

        """
        Contributions to the transfinite boundary interpolant from this vertex
        """
        term_a = boundary_function(i, x_ip1, y_ip1)
        term_b = boundary_function(im1, x_im1, y_im1)
        term_c = boundary_function(i, vertices[:, i, 0], vertices[:, i, 1])
        g += lambdas[:, :, i] * (term_a + term_b - term_c[:, None])

    return g

def main():

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
                                         uniform_boundary = uniform_boundary,
                                         uniform_rescale = uniform_scale,
                                         accumulating_power = accumulating_power,
                                         accumulating_border = accumulating_border,
                                         polygon = num_vertices > 3)

    num_points_per_polygon = reference_nodes.shape[1] * num_vertices
    points = np.zeros([vertices.shape[0], num_points_per_polygon, 2])
    for c in range(vertices.shape[0]):

        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(vertices[c, :, :], internal_point[c, :, :])
        triangles_points = geometry_utilities.extract_triangulation_points_by_internal_point(vertices[c, :, :], internal_point[c, :, :], list_triangles)

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
        local_vertices = vertices[c, :, :].numpy().T

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # 1 row, 3 columns

        num_vertices = local_vertices.shape[1]
        coord = np.zeros([2, num_vertices + 1])
        coord[:, 0:num_vertices] = local_vertices
        coord[:, num_vertices] = local_vertices[:, 0]
        ax.plot(coord[0, :], coord[1, :], label='Polygon', linewidth=3.0, markersize=12)
        ax.scatter(points[c, :, 0], points[c, :, 1], color='r', marker='o', label='Points')

        plt.show()

        for v in range(vertices.shape[1]):
            # plot 3D surface
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')

            sc = ax.scatter(
                points[c, :, 0],
                points[c, :, 1],
                lambdas[c, :, v],
                c=lambdas[c, :, v],
                cmap='viridis',
                s=40
            )
            plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f" + str(v))

            plt.show()

        print("\nPartition of unity:")
        print(tf.reduce_sum(lambdas[c, :, :], axis=1).numpy())


if __name__ == "__main__":
    main()