from pypolydim import gedim
from src.NAVEM.border_sampler import *
from abc import ABC, abstractmethod
from typing import Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt

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
                                         function: Function, v_id: int = 0) -> Tuple[float, float]:

    distribution = PointsDistributionType.uniform
    polygon_vertices = function.domain_vertices()
    num_vertices = polygon_vertices.shape[1]

    err_points_on_edges, err_labels, _ = dataset_on_polygonal_border(polygon_vertices,
                                                                     n_err_pts, distribution, v_id)

    predicted_labels = np.squeeze(function.vander(err_points_on_edges), axis=1)
    err_max_l2 = float(np.linalg.norm(predicted_labels - err_labels, ord=np.inf))

    points_on_edges_nv, _, num_points_on_edge_nv, der_labels \
        = (dataset_on_polygonal_border_not_including_vertices(polygon_vertices,
                                                              n_err_pts, distribution, v_id))

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

    err_max_h1 = float(np.linalg.norm(np.squeeze(d_tangent_predicted_labels_interpolation) - der_labels, ord=np.inf))

    return err_max_l2, err_max_h1


def plot_points_distributions(function: Function, flat_poles: NDArray[np.float64], boundary_points: NDArray[np.float64]):
    ##################################################################
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()

    polygon_vertices = function.domain_vertices()
    num_vertices = polygon_vertices.shape[1]

    coord = np.zeros([3, num_vertices + 1])
    coord[:, 0:num_vertices] = polygon_vertices
    coord[:, num_vertices] = polygon_vertices[:, 0]
    ax.plot(coord[0, :], coord[1, :], label='vertices')
    #################################################################

    ##################################################################
    ax.scatter(flat_poles[0, :], flat_poles[1, :], label='poles')
    ##################################################################

    ##################################################################
    ax.scatter(boundary_points[0, :], boundary_points[1, :], label='boundary_points')
    ##################################################################

    ##################################################################
    plt.axis('equal')
    plt.legend()
    plt.show()
    ##################################################################


def plot_function_and_tangent_derivatives_on_edges(geometry_utilities: gedim.GeometryUtilities, n_err_pts, function, v_id=0):

    distribution = PointsDistributionType.uniform

    polygon_vertices = function.domain_vertices()
    num_vertices = polygon_vertices.shape[1]

    points_on_edges, true_labels, _ = (dataset_on_polygonal_border(polygon_vertices,
                                                                   n_err_pts, distribution, v_id))

    predicted_labels = np.squeeze(function.vander(points_on_edges), axis=1)

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

    points_on_edges_nv, _, num_points_on_edge_nv, _ \
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

    ##################################################################

    plt.legend()
    plt.show()
    ##################################################################


def plot_function_on_edges(geometry_utilities: gedim.GeometryUtilities, n_err_pts, function, v_id=0):

    distribution = "uniform"
    points_on_edges, true_labels, _ = (dataset_on_polygonal_border(function.polygon_vertices,
                                                                                  n_err_pts, distribution, v_id))

    predicted_labels = np.squeeze(function.vander(points_on_edges), axis=1)

    ##################################################################
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot(1, 2, 1, projection='3d')

    coord = np.zeros([3, function.num_vertices + 1])
    coord[:, 0:function.num_vertices] = function.polygon_vertices
    coord[:, function.num_vertices] = function.polygon_vertices[:, 0]
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

    plt.legend()
    plt.show()
    ##################################################################


def plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities: gedim.GeometryUtilities,
                                                             polygon_vertices,
                                                             n_err_pts, function,
                                                             internal_angles,
                                                             vertex_distance):

    distribution = "uniform"
    points_on_edges, _, _ = (dataset_on_polygonal_border(function.polygon_vertices,
                                                                        n_err_pts, distribution))

    points_on_edges_nv, _, num_points_on_edge_nv, _ \
        = (dataset_on_polygonal_border_not_including_vertices(function.polygon_vertices,
                                                                             n_err_pts, distribution))

    predicted_labels = np.squeeze(function.vander(points_on_edges), axis=1)
    num_phisical_vertices = polygon_vertices.shape[1]

    ##################################################################
    fig = plt.figure(figsize=plt.figaspect(0.5))
    #################################################################

    for v in range(num_phisical_vertices):

        translation_1, translation_2, B, B_inv = compute_map_f(geometry_utilities, v, polygon_vertices, function.polygon_vertices, internal_angles, vertex_distance)

        mapped_reference_vertices, _ = map_f(translation_1, translation_2, B, B_inv, function.polygon_vertices)

        mapped_points_on_edges, _ = map_f(translation_1, translation_2, B, B_inv, points_on_edges)

        mapped_points_on_edges_nv, _ = map_f(translation_1, translation_2, B, B_inv, points_on_edges_nv)

        ##################################################################
        ax = fig.add_subplot(int(np.floor(num_phisical_vertices * 0.5)), int(np.ceil(num_phisical_vertices * 0.5)),
                             v + 1, projection='3d')

        coord = np.zeros([3, polygon_vertices.shape[1] + 1])
        coord[:, 0:polygon_vertices.shape[1]] = polygon_vertices
        coord[:, polygon_vertices.shape[1]] = polygon_vertices[:, 0]
        ax.plot(coord[0, :], coord[1, :], coord[2, :], label='vertices')
        #################################################################

        ##################################################################
        coord = np.zeros([3, function.num_vertices + 1])
        coord[:, 0:function.num_vertices] = mapped_reference_vertices
        coord[:, function.num_vertices] = mapped_reference_vertices[:, 0]
        ax.plot(coord[0, :], coord[1, :], coord[2, :], label='hang vertices')
        #################################################################

        #################################################################
        ax.scatter(mapped_points_on_edges[0, :],
                   mapped_points_on_edges[1, :],
                   predicted_labels,
                   label='predicted labels')
        #################################################################

        jac = B_inv.T
        d_predicted_labels = compute_vander_derivatives(points_on_edges_nv, jac, function)

        dx_predicted_labels = d_predicted_labels[0, :, :]
        dy_predicted_labels = d_predicted_labels[1, :, :]

        d_tangent_predicted_labels_interpolation = np.zeros([points_on_edges_nv.shape[1], 1])
        for v in range(function.num_vertices):
            tangent = geometry_utilities.compute_segment_tangent(mapped_reference_vertices[:, v],
                                                                 mapped_reference_vertices[:,
                                                                 (v + 1) % function.num_vertices])
            tangent = tangent / np.linalg.norm(tangent)

            d_tangent_predicted_labels_interpolation[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] \
                = (dx_predicted_labels[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] * tangent[0] +
                   dy_predicted_labels[v * num_points_on_edge_nv:(v + 1) * num_points_on_edge_nv, :] * tangent[1])

        ax.scatter(mapped_points_on_edges_nv[0, :],
                   mapped_points_on_edges_nv[1, :],
                   np.squeeze(d_tangent_predicted_labels_interpolation),
                   label='tangent derivatives')

    ##################################################################

    plt.legend()
    plt.show()
    ##################################################################


def compute_vander_derivatives(original_points, jac, function):

    grad_hang_vandermonde = function.vander_derivatives(original_points)

    grad_vandermonde = np.zeros(grad_hang_vandermonde.shape)

    grad_vandermonde[0, :, :] = (jac[0, 0] * grad_hang_vandermonde[0, :, :]
                                 + jac[0, 1] * grad_hang_vandermonde[1, :, :])
    grad_vandermonde[1, :, :] = (jac[1, 0] * grad_hang_vandermonde[0, :, :]
                                 + jac[1, 1] * grad_hang_vandermonde[1, :, :])

    return grad_vandermonde

def map_vander_derivatives(grad_hang_vandermonde, jac):

    grad_vandermonde = np.zeros(grad_hang_vandermonde.shape)

    grad_vandermonde[0, :, :] = (jac[0, 0] * grad_hang_vandermonde[0, :, :]
                                 + jac[0, 1] * grad_hang_vandermonde[1, :, :])
    grad_vandermonde[1, :, :] = (jac[1, 0] * grad_hang_vandermonde[0, :, :]
                                 + jac[1, 1] * grad_hang_vandermonde[1, :, :])

    return grad_vandermonde


# Map F : reference -> polygon
def compute_map_f(geometry_utilities: gedim.GeometryUtilities,
                  v_id: int, vertices: np.ndarray, reference_vertices: np.ndarray,
                  internal_angles, vertex_distance):

    assert (vertices.shape[0] == 3 and vertices.shape[1] >= 3)
    num_vertices = vertices.shape[1]

    vert_after = vertices[:, (v_id + 1) % num_vertices]
    interior_angle = internal_angles[0, v_id]

    _, tangent = geometry_utilities.compute_exterior_bisector_direction(interior_angle,
                                                                        vertices[:, v_id],
                                                                        vert_after)


    argument = tangent[0]

    # to avoid +-1.000000000000002
    if argument < -1.0 + geometry_utilities.geometry_tol_1D:
        argument = -1.0
    elif argument > 1.0 - geometry_utilities.geometry_tol_1D:
        argument = 1.0

    # counterclockwise-wise rotation
    theta = np.arccos(argument)

    if tangent[1] < -geometry_utilities.geometry_tol_1D:
        theta = -theta  # clockwise rotation

    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])

    scaling = vertex_distance[v_id]

    assert np.linalg.det(rotation_matrix) > 0

    # Compute transformation matrices
    B = scaling * rotation_matrix
    B_inv = (1.0/scaling) * rotation_matrix.T

    # Compute translations
    translation_1 = -reference_vertices[:, 0:1]
    translation_2 = vertices[:, v_id:v_id+1]

    return translation_1, translation_2, B, B_inv


def map_f_inv(translation_1, translation_2, B, B_inv, points: np.ndarray):

    mapped_points = B_inv @ (points - translation_2) - translation_1

    return mapped_points, B.T


def map_f(translation_1, translation_2, B, B_inv, points: np.ndarray):

    mapped_points = B @ (points + translation_1) + translation_2

    return mapped_points, B_inv.T
