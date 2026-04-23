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

def map_pts_from_1d_to_2d(abscissa: NDArray[np.float64], edge_start: NDArray[np.float64], edge_end: NDArray[np.float64]) -> NDArray[np.float64]:

    mapped_points_2d = edge_start + abscissa * (edge_end - edge_start)
    return mapped_points_2d

def dataset_on_polygonal_border(vertices: NDArray[np.float64], npts: int,
                                distribution_type: PointsDistributionType) -> NDArray[np.float64]:

    num_vertices = vertices.shape[1]
    n_pts_on_edge = int(np.ceil(npts / num_vertices))

    ref_xy_standard = chebyshev_lobatto_nodes(0.0, 1.0, n_pts_on_edge)
    ref_xy_near_singularity = reference_points_distribution(0.0, 1.0, n_pts_on_edge, distribution_type)

    pts = np.zeros((3, n_pts_on_edge * num_vertices))

    pts[:, :n_pts_on_edge] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, 1:2] - vertices[:, 0:1])
    for v in range(1, num_vertices-1):
        pts[:, v*n_pts_on_edge:(v+1)*n_pts_on_edge] = vertices[:, v:v+1] + ref_xy_standard * (vertices[:, v+1:v+2] - vertices[:, v:v+1])
    pts[:, -n_pts_on_edge:] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, -1:] - vertices[:, 0:1])

    return pts

def dataset_on_polygonal_border_not_including_vertices(vertices: NDArray[np.float64], npts: int,
                                                       distribution_type: PointsDistributionType) -> NDArray[np.float64]:
    num_vertices = vertices.shape[1]
    n_pts_on_edge = int(np.ceil(npts / num_vertices))

    ref_xy_standard = chebyshev_lobatto_nodes(0.0, 1.0, n_pts_on_edge + 2)[1:-1]
    ref_xy_near_singularity = reference_points_distribution(0.0, 1.0, n_pts_on_edge + 2, distribution_type)[1:-1]

    pts = np.zeros((3, n_pts_on_edge * num_vertices))

    pts[:, :n_pts_on_edge] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, 1:2] - vertices[:, 0:1])
    for v in range(1, num_vertices - 1):
        pts[:, v * n_pts_on_edge:(v + 1) * n_pts_on_edge] = vertices[:, v:v + 1] + ref_xy_standard * (
                    vertices[:, v + 1:v + 2] - vertices[:, v:v + 1])
    pts[:, -n_pts_on_edge:] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, -1:] - vertices[:, 0:1])

    return pts
