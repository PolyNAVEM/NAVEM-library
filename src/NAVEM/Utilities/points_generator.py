import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import use
import importlib
from enum import Enum
from typing import Tuple, List
from pypolydim import gedim

use("Qt5Agg")
importlib.reload(plt)

class PointsSegmentDistributionType(Enum):
    uniform = 1
    chebyshev_lobatto = 2
    asymmetricChebyshevLobatto = 3
    exponential = 4

class BorderType(Enum):
    all_borders = 1
    no_borders = 2
    only_accumulating_border = 3

class PointsTriangleDistributionType(Enum):
    gauss = 1
    uniform = 2
    accumulated = 3

def chebyshev_lobatto_nodes(a: float, b: float, n: int) -> NDArray[np.float64]:
    """
        returns n Chebyshev nodes inside the interval [a, b]
    """

    i = np.array(range(n))
    x = -np.cos(i * np.pi / (n - 1))  # nodes inside interval [-1,1]

    return 0.5 * (b - a) * x + 0.5 * (b + a)  # map nodes [a,b]


def reference_points_distribution(a: float, b: float, npts: int, distribution_type: PointsSegmentDistributionType) -> NDArray[np.float64]:

    match distribution_type:
        case PointsSegmentDistributionType.uniform:
            ref_xy_standard = np.linspace(0.0, 1.0, npts)
            return (b - a) * ref_xy_standard + a
        case PointsSegmentDistributionType.chebyshev_lobatto:
            ref_xy_standard = chebyshev_lobatto_nodes(0, 1.0, npts)
            return 0.5 * (b - a) * ref_xy_standard + 0.5 * (b + a)
        case PointsSegmentDistributionType.asymmetricChebyshevLobatto:
            ref_xy_near_singularity = chebyshev_lobatto_nodes(0, 2.0, 2 * npts)[:npts]
            return 0.5 * (b - a) * ref_xy_near_singularity + 0.5 * (b + a)
        case PointsSegmentDistributionType.exponential:
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
                                distribution_type: PointsSegmentDistributionType) -> Tuple[NDArray[np.float64], int]:

    num_vertices = vertices.shape[1]
    n_pts_on_edge = int(np.ceil(npts / num_vertices))

    ref_xy_standard = chebyshev_lobatto_nodes(0.0, 1.0, n_pts_on_edge)
    ref_xy_near_singularity = reference_points_distribution(0.0, 1.0, n_pts_on_edge, distribution_type)

    pts = np.zeros((3, n_pts_on_edge * num_vertices))

    pts[:, :n_pts_on_edge] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, 1:2] - vertices[:, 0:1])
    for v in range(1, num_vertices-1):
        pts[:, v*n_pts_on_edge:(v+1)*n_pts_on_edge] = vertices[:, v:v+1] + ref_xy_standard * (vertices[:, v+1:v+2] - vertices[:, v:v+1])
    pts[:, -n_pts_on_edge:] = vertices[:, 0:1] + ref_xy_near_singularity * (vertices[:, -1:] - vertices[:, 0:1])

    return pts, n_pts_on_edge

def dataset_on_polygonal_border_not_including_vertices(vertices: NDArray[np.float64], npts: int,
                                                       distribution_type: PointsSegmentDistributionType) -> Tuple[NDArray[np.float64], int]:
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

    return pts, n_pts_on_edge

def plot_points_boundary_distributions(polygon_vertices, flat_poles: NDArray[np.float64], boundary_points: NDArray[np.float64]):
    ##################################################################
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()

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

def plot_grid_over_polygon(polygon_vertices: NDArray[np.float64], points: NDArray[np.float64], filename: str=None) -> None:
    ##################################################################
    if filename is None:
        fig = plt.figure(figsize=plt.figaspect(0.5))
    else:
        fig = plt.figure()
    #################################################################

    #################################################################
    ax = fig.add_subplot()
    coord = np.zeros([3, polygon_vertices.shape[1] + 1])
    coord[:, 0:polygon_vertices.shape[1]] = polygon_vertices
    coord[:, polygon_vertices.shape[1]] = polygon_vertices[:, 0]
    ax.plot(coord[0, :], coord[1, :])
    #################################################################


    #################################################################
    ax.scatter(points[0, :],
               points[1, :],
               s=15,
               c='red')
    #################################################################

    plt.axis('equal')

    if filename is not None:
        plt.axis('off')
        plt.savefig(filename+".png", bbox_inches='tight', transparent=True, pad_inches=0)
    # else:
    #     plt.axis('equal')
    plt.show()
    #################################################################

def uniform_grid_over_triangle(m: int, boundary: bool = False, rescale: bool = False, polygon: bool = False) -> NDArray[np.float64]:

    # polygon = True if and only if this function is called from "uniform_grid_over_polygon"
    if boundary:
        if polygon:
            n_points = int((m + 1) * (m + 2) * 0.5) - (m + 1)
            s = np.linspace(0.0, 1.0, m + 1)
        else:
            n_points = int((m + 1) * (m + 2) * 0.5)
            s = np.linspace(0.0, 1.0, m + 1)
    else:
        n_points = int((m - 1) * (m - 2) * 0.5)
        if rescale:
            d = 1.0 / (1.0 + m)
            s = np.linspace(-d * 0.5, 1.0 + d * 0.5, m + 1)[1:-2]
        else:
            s = np.linspace(0.0, 1.0, m + 1)[1:-2]

    xy_points = np.zeros([3, n_points])

    id_pt = 0
    for i in range(len(s)):

        if polygon and boundary and i == 0:
            continue

        for j in range(len(s) - i):
            xy_points[0, id_pt] = s[j]
            xy_points[1, id_pt] = s[i]
            id_pt += 1

    return xy_points



def concentrating_grid_over_triangle(n: int, power: float, border_type: BorderType) -> NDArray[np.float64]:

    """

    Args:
        n: number of points
        power: how points are dense near accumulating edge: the gratest the more the points are accumulated
        border_type:

    Returns: points (3 x num_points)

    """

    points = []

    for i in range(n + 1):
        for j in range(n + 1 - i):

            l1 = 0
            l2 = 0
            l3 = 0

            match border_type:
                case BorderType.all_borders:
                    k = n - i - j
                    l1 = i / n
                    l2 = j / n
                    l3 = k / n  # l3 = 1 - l1 - l2
                case BorderType.no_borders:
                    k = n - i - j
                    l1 = (i + 0.5) / (n + 1.5)
                    l2 = (j + 0.5) / (n + 1.5)
                    l3 = (k + 0.5) / (n + 1.5)
                case BorderType.only_accumulating_border:
                    k = n - i - j - 0.5
                    l1 = (i + 0.5) / (n + 1)
                    l2 = (j + 0.5) / (n + 1)
                    l3 = (k + 0.5) / (n + 1)
                case _:
                    raise ValueError("invalid border type")

            # Apply bias only to lambda3 (C vertex = (0,1) )
            l3_biased = (1 - np.cos(np.pi * l3 / 2)) ** power

            scale = (1 - l3_biased) / (1 - l3 + 1e-14)
            l1_biased = l1 * scale
            l2_biased = l2 * scale

            x = l1_biased
            y = l2_biased
            points.append((x, y, 0))

    points = np.array(points).T
    return points


def grid_over_triangle(points_distribution_type: PointsTriangleDistributionType, quadrature_order: int,
                       uniform_boundary: bool = False, uniform_rescale: bool = True,
                       accumulating_power: float = 1.0, accumulating_border: BorderType = BorderType.no_borders,
                       polygon: bool = True) -> NDArray[np.float64]:

    match points_distribution_type:
        case PointsTriangleDistributionType.gauss:
            quadrature = gedim.quadrature.Quadrature_Gauss2D_Triangle.fill_points_and_weights(quadrature_order)
            return quadrature.points
        case PointsTriangleDistributionType.uniform:
            n_points_per_pol = int((quadrature_order + 1) * (quadrature_order + 0) * 0.5)
            return uniform_grid_over_triangle(n_points_per_pol, boundary=uniform_boundary,
                                              rescale=uniform_rescale, polygon=polygon)
        case PointsTriangleDistributionType.accumulated:
            return concentrating_grid_over_triangle(quadrature_order, accumulating_power, accumulating_border)
        case _:
            raise ValueError("invalid points triangle distribution type")

def uniform_grid_over_polygon(reference_points: NDArray[np.float64],
                              polygon_triangulation_by_interior_points: List[NDArray[np.float64]],
                              boundary: bool = False) -> NDArray[np.float64]:

    # Triangulation by interior point
    num_triangles = len(polygon_triangulation_by_interior_points)
    polygon: bool = (num_triangles > 1)

    num_quadrature = reference_points.shape[1]
    if polygon and boundary:
        nodes = np.zeros([3, num_quadrature * num_triangles + 1], dtype=float)
    else:
        nodes = np.zeros([3, num_quadrature * num_triangles], dtype=float)

    map_triangle = gedim.MapTriangle()
    for idx_t in range(num_triangles):
        triangle_vertices = polygon_triangulation_by_interior_points[idx_t]
        map_data: gedim.MapTriangle.MapTriangleData = map_triangle.compute(triangle_vertices)

        nodes[:, (idx_t * num_quadrature): ((idx_t + 1) * num_quadrature)] = map_triangle.f(map_data, reference_points)

        if idx_t == num_triangles - 1 and polygon and boundary:
            nodes[:, -1:] = map_triangle.f(map_data, np.zeros([3, 1]))

    return nodes


def concentrating_grid_over_polygon(reference_points: NDArray[np.float64],
                                    polygon_triangulation_by_interior_points: List[NDArray[np.float64]],
                                    border_type: BorderType) -> NDArray[np.float64]:

    # Triangulation by interior point
    num_triangles = len(polygon_triangulation_by_interior_points)

    num_quadrature = reference_points.shape[1]
    nodes = np.zeros([3, num_quadrature * num_triangles], dtype=float)

    map_triangle = gedim.MapTriangle()
    for idx_t in range(num_triangles):
        triangle_vertices = polygon_triangulation_by_interior_points[idx_t]
        map_data: gedim.MapTriangle.MapTriangleData = map_triangle.compute(triangle_vertices)

        nodes[:, (idx_t * num_quadrature): ((idx_t + 1) * num_quadrature)] = map_triangle.f(map_data, reference_points)

    if border_type == BorderType.all_borders:
        nodes = np.unique(nodes, axis=1)

    return nodes


def gauss_points_over_polygon(reference_points: NDArray[np.float64],
                              polygon_triangulation: List[NDArray[np.float64]]):

    num_quadrature = reference_points.shape[1]
    num_triangles = len(polygon_triangulation)

    nodes = np.zeros([3, num_quadrature * num_triangles], dtype=float)

    map_triangle = gedim.MapTriangle()
    for idx_t in range(num_triangles):
        triangle_vertices = polygon_triangulation[idx_t]
        map_data: gedim.MapTriangle.MapTriangleData = map_triangle.compute(triangle_vertices)

        nodes[:, (idx_t * num_quadrature): ((idx_t + 1) * num_quadrature)] = map_triangle.f(map_data, reference_points)

    return nodes

def grid_over_polygon(points_distribution_type: PointsTriangleDistributionType,
                      reference_points: NDArray[np.float64],
                      polygon_triangulation_by_interior_points: List[NDArray[np.float64]],
                      uniform_boundary: bool = False, accumulating_border: BorderType = BorderType.no_borders) -> NDArray[np.float64]:

    match points_distribution_type:
        case PointsTriangleDistributionType.gauss:
            return gauss_points_over_polygon(reference_points, polygon_triangulation_by_interior_points)
        case PointsTriangleDistributionType.uniform:
            return uniform_grid_over_polygon(reference_points,
                                             polygon_triangulation_by_interior_points,
                                             uniform_boundary)
        case PointsTriangleDistributionType.accumulated:
            return concentrating_grid_over_polygon(reference_points, polygon_triangulation_by_interior_points, accumulating_border)
        case _:
            raise ValueError("invalid points triangle distribution type")
