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

from NAVEM.Utilities.points_generator import *
from abc import ABC, abstractmethod
from typing import Callable
import matplotlib.pyplot as plt
from NAVEM.geometry.geometry_utilities import *


class Function(ABC):

    @abstractmethod
    def vander(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def vander_derivatives(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def domain_vertices(self) -> NDArray[np.float64]:
        pass


def evaluate_accuracy_on_domain_boundary(geometry_utilities: gedim.GeometryUtilities, n_err_pts: int,
                                         function: Function,
                                         boundary_dirichlet_conditions: Callable[
                                             [int, NDArray[np.float64]], NDArray[np.float64]],
                                         boundary_tangent_derivatives: Callable[
                                             [int, NDArray[np.float64], NDArray[np.float64]], NDArray[
                                                 np.float64]] = None) -> Tuple[float, float]:
    distribution = PointsSegmentDistributionType.uniform
    polygon_vertices = function.domain_vertices()
    num_vertices = polygon_vertices.shape[1]

    err_points_on_edges, n_pts_on_edge = dataset_on_polygonal_border(polygon_vertices,
                                                                     n_err_pts,
                                                                     distribution)

    err_labels = np.zeros(n_pts_on_edge * num_vertices)
    for v in range(num_vertices):
        err_labels[v * n_pts_on_edge: (v + 1) * n_pts_on_edge] \
            = boundary_dirichlet_conditions(v, err_points_on_edges[:, v * n_pts_on_edge: (v + 1) * n_pts_on_edge])

    predicted_labels = np.squeeze(function.vander(err_points_on_edges), axis=1)
    err_max_l2 = float(np.linalg.norm(predicted_labels - err_labels, ord=np.inf))

    err_max_h1 = np.nan
    if boundary_tangent_derivatives:
        points_on_edges_nv, num_points_on_edge_nv \
            = (dataset_on_polygonal_border_not_including_vertices(polygon_vertices,
                                                                  n_err_pts, distribution))

        der_labels = np.zeros(num_points_on_edge_nv * num_vertices)
        for v in range(num_vertices):
            der_labels[v * num_points_on_edge_nv: (v + 1) * num_points_on_edge_nv] \
                = boundary_tangent_derivatives(v,
                                               polygon_vertices,
                                               err_points_on_edges[:, v * num_points_on_edge_nv:(v + 1) *
                                                                   num_points_on_edge_nv])

        d_predicted_labels = function.vander_derivatives(points_on_edges_nv)
        dx_predicted_labels = d_predicted_labels[0, :, :]
        dy_predicted_labels = d_predicted_labels[1, :, :]

        d_tangent_predicted_labels_interpolation = np.zeros([points_on_edges_nv.shape[1], 1])
        for v in range(num_vertices):
            tangent = geometry_utilities.segment_tangent(polygon_vertices[:, v],
                                                         polygon_vertices[:, (v + 1) % num_vertices])
            tangent = tangent / np.linalg.norm(tangent)

            d_tangent_predicted_labels_interpolation[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] \
                = (dx_predicted_labels[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] * tangent[0] +
                   dy_predicted_labels[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] * tangent[1])

        err_max_h1 = float(
            np.linalg.norm(np.squeeze(d_tangent_predicted_labels_interpolation) - der_labels, ord=np.inf))

    return err_max_l2, err_max_h1


def plot_function_and_tangent_derivatives_on_edges(geometry_utilities: gedim.GeometryUtilities, n_err_pts, function,
                                                   boundary_dirichlet_conditions: Callable[
                                                       [int, NDArray[np.float64]], NDArray[np.float64]],
                                                   boundary_tangent_derivatives: Callable[
                                                       [int, NDArray[np.float64], NDArray[np.float64]], NDArray[
                                                           np.float64]] = None) -> None:
    distribution = PointsSegmentDistributionType.uniform

    polygon_vertices = function.domain_vertices()
    num_vertices = polygon_vertices.shape[1]

    points_on_edges, n_pts_on_edge = dataset_on_polygonal_border(polygon_vertices, n_err_pts, distribution)

    predicted_labels = np.squeeze(function.vander(points_on_edges), axis=1)

    true_labels = np.zeros(n_pts_on_edge * num_vertices)
    for v in range(num_vertices):
        true_labels[v * n_pts_on_edge:
                    (v+1) * n_pts_on_edge] = boundary_dirichlet_conditions(v, points_on_edges[:, v * n_pts_on_edge:
                                                                                              (v+1) * n_pts_on_edge])

    ##################################################################
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot(1, 2, 1, projection='3d')

    coord = np.zeros([3, num_vertices + 1])
    coord[:, 0:num_vertices] = polygon_vertices
    coord[:, num_vertices] = polygon_vertices[:, 0]
    ax.plot(coord[0, :], coord[1, :], coord[2, :], label='vertices')
    #################################################################

    ##################################################################
    ax.scatter(points_on_edges[0, :],
               points_on_edges[1, :],
               true_labels,
               label='true labels')

    ax.scatter(points_on_edges[0, :],
               points_on_edges[1, :],
               predicted_labels,
               label='predicted labels')
    ##################################################################

    points_on_edges_nv, num_points_on_edge_nv \
        = (dataset_on_polygonal_border_not_including_vertices(polygon_vertices,
                                                              n_err_pts, distribution))

    d_predicted_labels = function.vander_derivatives(points_on_edges_nv)
    dx_predicted_labels = d_predicted_labels[0, :, :]
    dy_predicted_labels = d_predicted_labels[1, :, :]

    d_tangent_predicted_labels_interpolation = np.zeros([points_on_edges_nv.shape[1], 1])
    for v in range(num_vertices):
        tangent = geometry_utilities.segment_tangent(polygon_vertices[:, v],
                                                     polygon_vertices[:,
                                                     (v + 1) % num_vertices])
        tangent = tangent / np.linalg.norm(tangent)

        d_tangent_predicted_labels_interpolation[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] \
            = (dx_predicted_labels[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] * tangent[0] +
               dy_predicted_labels[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] * tangent[1])

    ax.scatter(points_on_edges_nv[0, :],
               points_on_edges_nv[1, :],
               np.squeeze(d_tangent_predicted_labels_interpolation),
               label='tangent derivatives')

    if boundary_tangent_derivatives is not None:
        der_labels = np.zeros(num_points_on_edge_nv * num_vertices)
        for v in range(num_vertices):
            der_labels[v * num_points_on_edge_nv: (v + 1) * num_points_on_edge_nv] \
                = boundary_tangent_derivatives(v, polygon_vertices,
                                               points_on_edges_nv[:, v * num_points_on_edge_nv:
                                                                  (v + 1) * num_points_on_edge_nv])

        ax.scatter(points_on_edges_nv[0, :],
                   points_on_edges_nv[1, :],
                   der_labels,
                   label='true tangent derivatives')

    ##################################################################

    plt.legend()
    plt.show()
    ##################################################################


def map_vander_derivatives(grad_hang_vandermonde: NDArray[np.float64], jac: NDArray[np.float64]) -> NDArray[np.float64]:
    grad_vandermonde = np.zeros(grad_hang_vandermonde.shape)

    grad_vandermonde[0, :, :] = (jac[0, 0] * grad_hang_vandermonde[0, :, :]
                                 + jac[0, 1] * grad_hang_vandermonde[1, :, :])
    grad_vandermonde[1, :, :] = (jac[1, 0] * grad_hang_vandermonde[0, :, :]
                                 + jac[1, 1] * grad_hang_vandermonde[1, :, :])

    return grad_vandermonde


# Map F : reference -> polygon
def compute_map_f(geometry_utilities: gedim.GeometryUtilities,
                  v_id: int, vertices: NDArray[np.float64], reference_vertices: NDArray[np.float64],
                  internal_angles: List[float], vertex_distance: List[float]) \
        -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    assert (vertices.shape[0] == 3 and vertices.shape[1] >= 3)
    num_vertices = vertices.shape[1]

    vert_after = vertices[:, (v_id + 1) % num_vertices]
    interior_angle = internal_angles[v_id]

    _, tangent = compute_exterior_bisector_direction(interior_angle,
                                                     vertices[:, v_id],
                                                     vert_after)

    argument = tangent[0]

    # to avoid +-1.000000000000002
    if argument < -1.0 + geometry_utilities.tolerance1_d():
        argument = -1.0
    elif argument > 1.0 - geometry_utilities.tolerance1_d():
        argument = 1.0

    # counterclockwise-wise rotation
    theta = np.arccos(argument)

    if tangent[1] < -geometry_utilities.tolerance1_d():
        theta = -theta  # clockwise rotation

    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])

    scaling = vertex_distance[v_id]

    assert np.linalg.det(rotation_matrix) > 0

    # Compute transformation matrices
    b_matrix = scaling * rotation_matrix
    b_matrix_inv = (1.0 / scaling) * rotation_matrix.T

    # Compute translations
    translation_1 = -reference_vertices[:, 0:1]
    translation_2 = vertices[:, v_id:v_id + 1]

    return translation_1, translation_2, b_matrix, b_matrix_inv


def map_f_inv(translation_1, translation_2, b_matrix, b_matrix_inv, points: np.ndarray):
    mapped_points = b_matrix_inv @ (points - translation_2) - translation_1

    return mapped_points, b_matrix.T


def map_f(translation_1, translation_2, b_matrix, b_matrix_inv, points: np.ndarray):
    mapped_points = b_matrix @ (points + translation_1) + translation_2

    return mapped_points, b_matrix_inv.T
