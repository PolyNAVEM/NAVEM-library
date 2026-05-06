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

import numpy as np
import math
from typing import List, Tuple
from pypolydim import gedim, polydim
from numpy.typing import NDArray
from NAVEM.Utilities.NAVEMPolygon import NAVEMPolygon
from typing import Dict

class MeshGeometricData2D:

    def __init__(self, mesh_geometric_data: gedim.MeshUtilities.MeshGeometricData2D,
                 cell2_ds_list_triangles: List[List[int]], cell2_ds_polygon: List[NAVEMPolygon],
                 cell2_ds_mapped_polygon_internal_angles: List[List[float]]):

        self.mesh_geometric_data = mesh_geometric_data
        self.cell2_ds_list_triangles = cell2_ds_list_triangles
        self.cell2_ds_polygon = cell2_ds_polygon
        self.cell2_ds_mapped_polygon_internal_angles = cell2_ds_mapped_polygon_internal_angles


def polygon_is_simple(geometry_utilities: gedim.GeometryUtilities,
                      polygon_vertex_points: NDArray[np.float64]) -> bool:

    num_vertices: int = polygon_vertex_points.shape[1]

    if polygon_vertex_points.shape[0] != 3 or num_vertices < 3:
        raise ValueError("Not valid points in compute unaligned points")

    is_simple = True
    for e1 in range(num_vertices):

        prev_e1 = e1 - 1 if e1 > 0 else num_vertices - 1

        for e2 in np.arange(e1 + 2, num_vertices):

            if e2 == prev_e1:
                continue

            origin_1 = polygon_vertex_points[:, e1]
            end_1 = polygon_vertex_points[:, (e1 + 1) % num_vertices]
            origin_2 = polygon_vertex_points[:, e2]
            end_2 = polygon_vertex_points[:, (e2 + 1) % num_vertices]

            segment_intersection_data: gedim.GeometryUtilities.IntersectionSegmentSegmentResult \
                = geometry_utilities.intersection_segment_segment(origin_1, end_1, origin_2, end_2)

            match segment_intersection_data.intersection_segments_type:
                case gedim.GeometryUtilities.IntersectionSegmentSegmentResult.IntersectionSegmentTypes.no_intersection:
                    pass
                case gedim.GeometryUtilities.IntersectionSegmentSegmentResult.IntersectionSegmentTypes.single_intersection:
                    is_simple = False
                case gedim.GeometryUtilities.IntersectionSegmentSegmentResult.IntersectionSegmentTypes.multiple_intersections:
                    is_simple = False
                case _:
                    raise ValueError("Not a valid intersection segment type")

            if not is_simple:
                break

        if not is_simple:
            break

    return is_simple

def compute_geometric_properties_mesh_2(geometry_utilities: gedim.GeometryUtilities,
                                        mesh_utilities: gedim.MeshUtilities,
                                        mesh: gedim.MeshMatricesDAO) -> MeshGeometricData2D:

    mesh_geometric_data = polydim.pde_tools.mesh.pde_mesh_utilities.compute_mesh_2_d_geometry_data(geometry_utilities,
                                                                                                   mesh_utilities,
                                                                                                   mesh)

    cell2_ds_list_triangles = []
    cell2_ds_polygon = []
    cell2_ds_mapped_polygon_internal_angles = []

    for c in range(mesh.cell2_d_total_number()):

        vertex_points = mesh_geometric_data.cell2_ds_vertices[c]
        list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(vertex_points)
        cell2_ds_list_triangles.append(list_triangles)

        polygon = NAVEMPolygon(geometry_utilities, mesh_geometric_data.cell2_ds_vertices[c],
                               mesh_geometric_data.cell2_ds_diameters[c],
                               mesh_geometric_data.cell2_ds_centroids[c],
                               list_triangles)
        cell2_ds_polygon.append(polygon)

        for v1 in range(vertex_points.shape[1]):

            max_distance = 0.0
            first_vertex = polygon.mapped_vertices[:, v1]
            for v2 in range(vertex_points.shape[1]):

                if v1 != v2:
                    second_vertex = polygon.mapped_vertices[:, v2]
                    in_distance = np.linalg.norm(first_vertex - second_vertex)

                    if max_distance < in_distance:
                        max_distance = in_distance

            polygon.mapped_max_vertex_distance.append(max_distance)

        mapped_polygon_internal_angles = compute_polygon_interior_angles(polygon.mapped_vertices)
        cell2_ds_mapped_polygon_internal_angles.append(mapped_polygon_internal_angles)

    return MeshGeometricData2D(mesh_geometric_data, cell2_ds_list_triangles, cell2_ds_polygon, cell2_ds_mapped_polygon_internal_angles)


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

def compute_exterior_bisector_direction(interior_angle: float, v: NDArray[np.float64], v_next: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    # clockwise rotation
    theta = -(np.pi - interior_angle * 0.5)

    tangent = v_next - v  # center at the origin to correctly rotate

    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])

    tangent = rotation_matrix @ tangent
    tangent = tangent / np.linalg.norm(tangent)
    origin = v

    return origin, tangent

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

def z_cross_2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def compute_polygon_kernel(geometry_utilities: gedim.GeometryUtilities, vertices: NDArray[np.float64],
                           polygon_internal_angles: List[float]) -> NDArray[np.float64]:

    num_vertices = vertices.shape[1]

    if num_vertices == 3:
        return vertices

    assert num_vertices >= 4

    kernel = vertices
    for v in range(num_vertices):

        # concave angle
        edge_origin = vertices[:, v]
        edge_direction = vertices[:, (v + 1) % num_vertices] - vertices[:, v]

        if (polygon_internal_angles[v] > np.pi or
                polygon_internal_angles[(v + 1) % num_vertices] > np.pi):

            kernel_next = np.array([], dtype=np.float64).reshape(3, 0)
            num_vertices_actual_kernel = kernel.shape[1]

            if num_vertices_actual_kernel == 0:
                return np.zeros([3, 0])

            x = kernel[:, num_vertices_actual_kernel - 1]
            a = z_cross_2d(edge_direction, x - edge_origin) > -geometry_utilities.tolerance1_d()

            for j in np.arange(0, num_vertices_actual_kernel):

                # if the current vertex is on the right side w.r.t. lines, it is in the kernel

                x = kernel[:, j]
                b = z_cross_2d(edge_direction, x - edge_origin) > -geometry_utilities.tolerance1_d()

                if (not a and b) or (not b and a):

                    segment_origin = kernel[:, j - 1 if j > 0 else num_vertices_actual_kernel - 1]
                    segment_end = x
                    intersection_results: gedim.GeometryUtilities.IntersectionSegmentSegmentResult = (
                        geometry_utilities.intersection_segment_segment(segment_origin,
                                                                        segment_end, edge_origin, vertices[:, (v + 1) % num_vertices]))

                    if len(intersection_results.first_segment_intersections) == 1:

                        intersection: gedim.GeometryUtilities.IntersectionSegmentSegmentResult.IntersectionPosition = intersection_results.first_segment_intersections[0]
                        match intersection.type:
                            case gedim.GeometryUtilities.PointSegmentPositionTypes.inside_segment:
                                # TO DO: is the point already inserted?
                                points = segment_origin + intersection.curvilinear_coordinate * (segment_end - segment_origin)
                                kernel_next = np.concatenate([kernel_next, np.expand_dims(points, axis=1)], axis=1)
                            case _:
                                pass

                if b:
                    kernel_next = np.concatenate([kernel_next, np.expand_dims(x, axis=1)], axis=1)

                a = b

            kernel = kernel_next

    if kernel.shape[1] < 3:
        kernel = np.zeros([3, 0])

    return kernel


def select_non_convex_elements(mesh: gedim.MeshMatricesDAO,
                               mesh_geometric_data: MeshGeometricData2D,
                               tolerance_angle: float) -> Tuple[gedim.MeshMatrices, gedim.MeshMatricesDAO]:

    non_convex_mesh_data = gedim.MeshMatrices()
    non_convex_mesh = gedim.MeshMatricesDAO(non_convex_mesh_data)
    non_convex_mesh.initialize_dimension(2)

    for c in range(mesh.cell2_d_total_number()):

        internal_angles = mesh_geometric_data.cell2_ds_mapped_polygon_internal_angles[c]

        is_concave = any(a > np.pi * (1.0 + tolerance_angle) for a in internal_angles)

        if is_concave:

            num_vertices = mesh.cell2_d_number_vertices(c)
            id_new_vertices = non_convex_mesh.cell0_d_append(num_vertices)
            map_vertices: Dict[int, int] = {}
            for v in range(num_vertices):

                cell0_d_index = mesh.cell2_d_vertex(c, v)

                map_vertices[cell0_d_index] = id_new_vertices

                non_convex_mesh.cell0_d_insert_coordinates(id_new_vertices, mesh.cell0_d_coordinates(cell0_d_index))
                non_convex_mesh.cell0_d_set_marker(id_new_vertices, mesh.cell0_d_marker(cell0_d_index))
                non_convex_mesh.cell0_d_set_state(id_new_vertices, mesh.cell0_d_is_active(cell0_d_index))

                id_new_vertices += 1

            num_edges = mesh.cell2_d_number_edges(c)
            id_new_edges = non_convex_mesh.cell1_d_append(num_edges)
            map_edges: Dict[int, int] = {}
            for e in range(num_edges):
                cell1_d_index = mesh.cell2_d_edge(c, e)

                map_edges[cell1_d_index] = id_new_edges

                non_convex_mesh.cell1_d_insert_extremes(id_new_edges, map_vertices[mesh.cell1_d_origin(cell1_d_index)], map_vertices[mesh.cell1_d_end(cell1_d_index)])
                non_convex_mesh.cell1_d_set_marker(id_new_edges, mesh.cell1_d_marker(cell1_d_index))
                non_convex_mesh.cell1_d_set_state(id_new_edges, mesh.cell1_d_is_active(cell1_d_index))

                id_new_edges += 1

            id_new_cell = non_convex_mesh.cell2_d_append(1)

            non_convex_mesh.cell2_d_initialize_vertices(id_new_cell, num_vertices)
            for v in range(num_vertices):
                non_convex_mesh.cell2_d_insert_vertex(id_new_cell, v, map_vertices[mesh.cell2_d_vertex(c, v)])

            non_convex_mesh.cell2_d_initialize_edges(id_new_cell, num_edges)
            for e in range(num_edges):
                non_convex_mesh.cell2_d_insert_edge(id_new_cell, e, map_edges[mesh.cell2_d_edge(c, e)])

            non_convex_mesh.cell2_d_set_marker(id_new_cell, mesh.cell2_d_marker(c))
            non_convex_mesh.cell2_d_set_state(id_new_cell, mesh.cell2_d_is_active(c))

    return non_convex_mesh_data, non_convex_mesh

