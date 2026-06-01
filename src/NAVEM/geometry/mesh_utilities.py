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
from typing import List, Tuple
from pypolydim import gedim, polydim
from NAVEM.Utilities.NAVEMPolygon import NAVEMPolygon
from typing import Dict
from NAVEM.geometry.geometry_utilities import compute_polygon_interior_angles

class MeshGeometricData2D:

    def __init__(self, mesh_geometric_data: gedim.MeshUtilities.MeshGeometricData2D,
                 cell2_ds_list_triangles: List[List[int]], cell2_ds_polygon: List[NAVEMPolygon],
                 cell2_ds_mapped_polygon_internal_angles: List[List[float]],
                 cell2_ds_polygon_internal_angles: List[List[float]]):

        self.mesh_geometric_data = mesh_geometric_data
        self.cell2_ds_list_triangles = cell2_ds_list_triangles
        self.cell2_ds_polygon = cell2_ds_polygon
        self.cell2_ds_mapped_polygon_internal_angles = cell2_ds_mapped_polygon_internal_angles
        self.cell2_ds_polygon_internal_angles = cell2_ds_polygon_internal_angles

def compute_geometric_properties_mesh_2(geometry_utilities: gedim.GeometryUtilities,
                                        mesh_utilities: gedim.MeshUtilities,
                                        mesh: gedim.MeshMatricesDAO) -> MeshGeometricData2D:

    mesh_geometric_data = polydim.pde_tools.mesh.pde_mesh_utilities.compute_mesh_2_d_geometry_data(geometry_utilities,
                                                                                                   mesh_utilities,
                                                                                                   mesh)

    cell2_ds_list_triangles = []
    cell2_ds_polygon = []
    cell2_ds_mapped_polygon_internal_angles = []
    cell2_ds_polygon_internal_angles = []

    for c in range(mesh.cell2_d_total_number()):

        vertex_points = mesh_geometric_data.cell2_ds_vertices[c]
        polygon_internal_angles = compute_polygon_interior_angles(vertex_points)
        cell2_ds_polygon_internal_angles.append(polygon_internal_angles)

        list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(vertex_points)
        cell2_ds_list_triangles.append(list_triangles)

        polygon = NAVEMPolygon(geometry_utilities,
                               mesh_geometric_data.cell2_ds_vertices[c],
                               mesh_geometric_data.cell2_ds_diameters[c],
                               mesh_geometric_data.cell2_ds_centroids[c],
                               polygon_internal_angles)

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

    return MeshGeometricData2D(mesh_geometric_data, cell2_ds_list_triangles, cell2_ds_polygon, cell2_ds_mapped_polygon_internal_angles, cell2_ds_polygon_internal_angles)


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

