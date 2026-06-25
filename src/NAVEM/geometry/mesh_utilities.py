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
from NAVEM.geometry.geometry_utilities import compute_polygon_interior_angles, compute_polygon_kernel
from NAVEM.PCC_2D.NAVEM_Data_PCC_2D import NNCategory
from numpy.typing import NDArray


class MeshGeometricData2D:

    def __init__(self, mesh_geometric_data: gedim.MeshUtilities.MeshGeometricData2D,
                 cell2_ds_list_triangles: List[List[int]], cell2_ds_polygon: List[NAVEMPolygon],
                 cell2_ds_mapped_polygon_internal_angles: List[List[float]],
                 cell2_ds_polygon_internal_angles: List[List[float]],
                 cell2_ds_triangulations: List[List[NDArray[np.float64]]]):

        self.mesh_geometric_data = mesh_geometric_data
        self.cell2_ds_vertices = mesh_geometric_data.cell2_ds_vertices
        self.cell2_ds_list_triangles = cell2_ds_list_triangles
        self.cell2_ds_triangulations: List[List[NDArray[np.float64]]] = cell2_ds_triangulations
        self.cell2_ds_polygon = cell2_ds_polygon
        self.cell2_ds_mapped_polygon_internal_angles = cell2_ds_mapped_polygon_internal_angles
        self.cell2_ds_polygon_internal_angles = cell2_ds_polygon_internal_angles
        self.cell2_ds_centroids = mesh_geometric_data.cell2_ds_centroids
        self.cell2_ds_areas = mesh_geometric_data.cell2_ds_areas
        self.cell2_ds_diameters = mesh_geometric_data.cell2_ds_diameters
        self.cell2_ds_edge_lengths = mesh_geometric_data.cell2_ds_edge_lengths
        self.cell2_ds_edge_directions = mesh_geometric_data.cell2_ds_edge_directions
        self.cell2_ds_edge_tangents = mesh_geometric_data.cell2_ds_edge_tangents
        self.cell2_ds_edge_normals = mesh_geometric_data.cell2_ds_edge_normals

def compute_geometric_properties_mesh_2(geometry_utilities: gedim.GeometryUtilities,
                                        mesh_utilities: gedim.MeshUtilities,
                                        mesh: gedim.MeshMatricesDAO) -> MeshGeometricData2D:

    mesh_geometric_data = polydim.pde_tools.mesh.pde_mesh_utilities.compute_mesh_2_d_geometry_data(geometry_utilities,
                                                                                                   mesh_utilities,
                                                                                                   mesh)

    cell2_ds_list_triangles = []
    cell2_ds_polygon = []
    cell2_ds_mapped_polygon_internal_angles = []
    cell2_ds_triangulations = []
    cell2_ds_polygon_internal_angles = []

    for c in range(mesh.cell2_d_total_number()):

        vertex_points = mesh_geometric_data.cell2_ds_vertices[c]

        polygon_internal_angles = compute_polygon_interior_angles(vertex_points)
        cell2_ds_polygon_internal_angles.append(polygon_internal_angles)


        if polygon_internal_angles is not None:
            use_kernel = NNCategory.is_concave(polygon_internal_angles)
        else:
            use_kernel = False

        if use_kernel:
            inertia_vertices = compute_polygon_kernel(geometry_utilities, vertex_points, polygon_internal_angles)
            kernal_area = geometry_utilities.polygon_area(inertia_vertices)
            internal_point = geometry_utilities.polygon_centroid(inertia_vertices, kernal_area)
            diameter = geometry_utilities.polygon_diameter(inertia_vertices)
        else:
            internal_point = mesh_geometric_data.cell2_ds_centroids[c]
            diameter = mesh_geometric_data.cell2_ds_diameters[c]
            inertia_vertices = vertex_points

        list_triangles_by_internal_point = geometry_utilities.polygon_triangulation_by_internal_point(vertex_points, internal_point)

        polygon = NAVEMPolygon(geometry_utilities,
                               vertex_points,
                               inertia_vertices,
                               internal_point,
                               diameter,
                               list_triangles_by_internal_point)

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

        list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(vertex_points)
        cell2_ds_list_triangles.append(list_triangles)

        triangulation_points = geometry_utilities.extract_triangulation_points(vertex_points, list_triangles)
        cell2_ds_triangulations.append(triangulation_points)

    return MeshGeometricData2D(mesh_geometric_data, cell2_ds_list_triangles, cell2_ds_polygon, cell2_ds_mapped_polygon_internal_angles, cell2_ds_polygon_internal_angles, cell2_ds_triangulations)


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

