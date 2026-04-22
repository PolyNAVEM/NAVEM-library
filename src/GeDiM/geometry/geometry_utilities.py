import numpy as np
import math
from typing import List
from pypolydim import gedim
from numpy.typing import NDArray


def compute_interior_angle(v: NDArray[np.float64],
                           v_prev: NDArray[np.float64],
                           v_next: NDArray[np.float64]) -> float:

    """
    Compute the angle spanned by (v_prev - v) and (v_next - v)

    Parameters:
    - v: vertex coordinate np.array([3,])
    - v_prev: vertex coordinate np.array([3,])
    - v_next: vertex coordinate np.array([3,])
    """

    direction_1 = v_prev - v
    direction_2 = v_next - v

    alpha = math.atan2(direction_2[0] * direction_1[1] - direction_2[1] * direction_1[0],
                       direction_2[0] * direction_1[0] + direction_2[1] * direction_1[1])

    alpha =  alpha + 2.0 * np.pi if alpha < 0 else alpha

    return alpha

def compute_polygon_interior_angles(vertices: NDArray[np.float64]) -> List[float]:

    """
    Compute polygon interior angles

    Parameters:
    - vertices: polygon vertices coordinates np.array([3,num vertices])
    """

    num_vertices = vertices.shape[1]
    angles = []
    for i in range(num_vertices):
        i_prev = i - 1 if i >= 1 else num_vertices - 1
        i_next = (i + 1) % num_vertices
        angles.append(compute_interior_angle(vertices[:, i], vertices[:, i_prev], vertices[:, i_next]))

    return angles

def compute_polygon_external_bisectors(geometry_utilities: gedim.GeometryUtilities, polygon_edge_normals: NDArray[np.float64]) -> NDArray[np.float64]:

    """
    Compute polygon external bisectors given the normal: the external bisectors is defined (up to normalization) as
    b = n1 + n2

    Parameters:
    - geometry_utilities
    - polygon_edge_normals: polygon edge outward unit normals np.array([3, num vertices])
    """

    num_vertices = polygon_edge_normals.shape[1]

    vertex_bisectors = np.zeros([3, num_vertices])
    for v in range(num_vertices):
        vertex_bisectors[:, v] = (polygon_edge_normals[:, v] +
                                  polygon_edge_normals[:, v - 1 if v >= 1 else num_vertices - 1])

        norm_values = np.linalg.norm(vertex_bisectors[:, v])

        assert geometry_utilities.is_value_positive(float(norm_values), geometry_utilities.tolerance1_d())

        vertex_bisectors[:, v] = vertex_bisectors[:, v] / norm_values

    return vertex_bisectors