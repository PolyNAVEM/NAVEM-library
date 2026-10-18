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


def compute_g_wachspress(vertices: tf.Tensor, lambdas: tf.Tensor, boundary_function) -> tf.Tensor:
    """
    Compute transfinite interpolant of Dirichlet boundary functions
    """
    num_polygons = vertices.shape[0]
    num_vertices = vertices.shape[1]
    g = tf.zeros([num_polygons, lambdas.shape[1]], dtype=tf.float64)

    for i in range(num_vertices):

        # get the other vertices for edges attached to vertex i
        ip1 = (i + 1) % num_vertices  # next vertex
        im1 = (i - 1) % num_vertices  # previous vertex

        # (x,y) coordinates where the boundary function will be evaluated
        x_ip1 = ((1.0 - lambdas[:, :, ip1]) * vertices[:, None, i, 0]
                 + lambdas[:, :, ip1] * vertices[:, None, ip1, 0])
        y_ip1 = ((1.0 - lambdas[:, :, ip1]) * vertices[:, None, i, 1]
                 + lambdas[:, :, ip1] * vertices[:, None, ip1, 1])

        x_im1 = ((1.0 - lambdas[:, :, im1]) * vertices[:, None, i, 0]
                 + lambdas[:, :, im1] * vertices[:, None, im1, 0])
        y_im1 = ((1.0 - lambdas[:, :, im1]) * vertices[:, None, i, 1]
                 + lambdas[:, :, im1] * vertices[:, None, im1, 1])

        # Contributions to the transfinite boundary interpolant from this vertex
        term_a_i = boundary_function(i, x_ip1, y_ip1)
        term_b_i = boundary_function(im1, x_im1, y_im1)

        term_c = boundary_function(i, vertices[:, i, 0], vertices[:, i, 1])
        """
        Contributions to the transfinite boundary interpolant from this vertex
        """
        g += lambdas[:, :, i] * (term_a_i + term_b_i - term_c[:, None])


    return g

