import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import use
import importlib
from enum import Enum

use("Qt5Agg")
importlib.reload(plt)



class PointsDistributionType(Enum):
    uniform = 1
    chebyshev_lobatto = 2
    asymmetricChebyshevLobatto = 3
    exponential = 4

def chebyshev_lobatto_nodes(a: float, b: float, n: int) -> NDArray[np.float64]:
    """
        returns n Chebyshev nodes inside the interval [a, b]
    """

    i = np.array(range(n))
    x = -np.cos(i * np.pi / (n - 1))  # nodes inside interval [-1,1]

    return 0.5 * (b - a) * x + 0.5 * (b + a)  # map nodes [a,b]


def reference_points_distribution(a: float, b: float, npts: int, distribution_type: PointsDistributionType) -> NDArray[np.float64]:

    match distribution_type:
        case PointsDistributionType.uniform:
            ref_xy_standard = np.linspace(0.0, 1.0, npts)
            return (b - a) * ref_xy_standard + a
        case PointsDistributionType.chebyshev_lobatto:
            ref_xy_standard = chebyshev_lobatto_nodes(0, 1.0, npts)
            return 0.5 * (b - a) * ref_xy_standard + 0.5 * (b + a)
        case PointsDistributionType.asymmetricChebyshevLobatto:
            ref_xy_near_singularity = chebyshev_lobatto_nodes(0, 2.0, 2 * npts)[:npts]
            return 0.5 * (b - a) * ref_xy_near_singularity + 0.5 * (b + a)
        case PointsDistributionType.exponential:
            scaling = 4.0
            base = np.exp(1.0)
            ref_xy_near_singularity = np.array(
                [base ** (-scaling * (np.sqrt(npts) - np.sqrt(i + 1))) for i in range(npts)])
            return (b - a) * ref_xy_near_singularity + a
        case _:
            raise ValueError("Choose a valid distribution")

def dataset_on_polygonal_border(vertices: NDArray[np.float64], npts: float,
                                distribution_type: PointsDistributionType, v_id: int = 0):

    num_vertices = vertices.shape[1]
    n_pts_on_edge = int(np.ceil(npts / num_vertices))

    ref_xy_standard = chebyshev_lobatto_nodes(0.0, 1.0, n_pts_on_edge)
    ref_xy_near_singularity = reference_points_distribution(0.0, 1.0, n_pts_on_edge, distribution_type)

    pts = np.zeros((3, n_pts_on_edge * num_vertices))
    labels = np.zeros(n_pts_on_edge * num_vertices)

    pts[:, :n_pts_on_edge] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, 1:2] - vertices[:, 0:1])
    for v in range(1, num_vertices-1):
        pts[:, v*n_pts_on_edge:(v+1)*n_pts_on_edge] = vertices[:, v:v+1] + ref_xy_standard * (vertices[:, v+1:v+2] - vertices[:, v:v+1])
    pts[:, -n_pts_on_edge:] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, -1:] - vertices[:, 0:1])

    if v_id == 0:
        labels[:n_pts_on_edge] = 1.0-ref_xy_near_singularity
        labels[-n_pts_on_edge:] = 1.0-ref_xy_near_singularity
    else:
        labels[v_id * n_pts_on_edge:(v_id + 1) * n_pts_on_edge] = 1.0 - ref_xy_near_singularity
        labels[(v_id - 1) * n_pts_on_edge:v_id * n_pts_on_edge] = ref_xy_near_singularity

    return pts, labels, n_pts_on_edge

def dataset_on_polygonal_border_not_including_vertices(vertices: np.ndarray, npts: float,
                                                       distribution_type: PointsDistributionType,
                                                       v_id: int = 0):
    num_vertices = vertices.shape[1]
    n_pts_on_edge = int(np.ceil(npts / num_vertices))

    ref_xy_standard = chebyshev_lobatto_nodes(0.0, 1.0, n_pts_on_edge + 2)[1:-1]
    ref_xy_near_singularity = reference_points_distribution(0.0, 1.0, n_pts_on_edge + 2, distribution_type)[1:-1]

    pts = np.zeros((3, n_pts_on_edge * num_vertices))
    labels = np.zeros(n_pts_on_edge * num_vertices)
    der_labels = np.zeros(n_pts_on_edge * num_vertices)

    pts[:, :n_pts_on_edge] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, 1:2] - vertices[:, 0:1])
    for v in range(1, num_vertices - 1):
        pts[:, v * n_pts_on_edge:(v + 1) * n_pts_on_edge] = vertices[:, v:v + 1] + ref_xy_standard * (
                    vertices[:, v + 1:v + 2] - vertices[:, v:v + 1])
    pts[:, -n_pts_on_edge:] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, -1:] - vertices[:, 0:1])

    if v_id == 0:
        labels[:n_pts_on_edge] = 1.0 - ref_xy_near_singularity
        der_labels[:n_pts_on_edge] = (-1.0/np.linalg.norm(vertices[:, 1:2] - vertices[:, 0:1])
                                      * np.ones(n_pts_on_edge))
        labels[-n_pts_on_edge:] = 1.0 - ref_xy_near_singularity
        der_labels[-n_pts_on_edge:] = (1.0 / np.linalg.norm(vertices[:, -1:] - vertices[:, 0:1])
                                       * np.ones(n_pts_on_edge))
    else:
        labels[v_id * n_pts_on_edge:(v_id + 1) * n_pts_on_edge] = 1.0 - ref_xy_near_singularity
        labels[(v_id - 1) * n_pts_on_edge:v_id * n_pts_on_edge] = ref_xy_near_singularity
        der_labels[v_id * n_pts_on_edge:(v_id + 1) * n_pts_on_edge] \
            = -1.0 / np.linalg.norm(vertices[:, 1:2] - vertices[:, 0:1]) * np.ones(
            n_pts_on_edge)
        der_labels[(v_id - 1) * n_pts_on_edge:v_id * n_pts_on_edge] \
            = 1.0 / np.linalg.norm(vertices[:, -1:] - vertices[:, 0:1]) * np.ones(
            n_pts_on_edge)
    return pts, labels, n_pts_on_edge, der_labels

def dataset_on_curved_border(n_pts: int, theta: float, radius: float):
    len_extern_perimeter = (2 * np.pi - theta) * radius
    pts_ext = int(np.ceil(len_extern_perimeter / (len_extern_perimeter + 2.0 * radius) * n_pts))
    pts_edg = int(np.ceil(0.5 * (n_pts - pts_ext)))
    tot_pts = pts_ext + 2 * pts_edg - 1
    half_theta = 0.5 * theta

    pts = np.zeros((tot_pts, 3))
    labels = np.zeros((tot_pts, 1))

    cheb_pts_1d = np.expand_dims(chebyshev_lobatto_nodes(0, 2.0, 2 * pts_edg)[:pts_edg], 1)
    pts[:pts_edg, :2] = cheb_pts_1d * np.array([np.cos(half_theta), np.sin(half_theta)])
    labels[:pts_edg] = 1.0 - cheb_pts_1d

    pts[pts_edg:2 * pts_edg - 1, :2] = cheb_pts_1d[1:] * np.array([np.cos(-half_theta), np.sin(-half_theta)])
    labels[pts_edg:2 * pts_edg - 1] = 1.0 - cheb_pts_1d[1:]

    angles = np.linspace(half_theta, 2 * np.pi - half_theta, pts_ext)
    pts[2 * pts_edg - 1:, :2] = np.array([np.cos(angles), np.sin(angles)]).T

    pts = pts * radius + np.array([1, 0, 0])
    return pts.T, np.squeeze(labels)

        