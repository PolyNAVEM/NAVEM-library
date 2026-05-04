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

import unittest
from NAVEM.geometry.geometry_utilities import *
import matplotlib.pyplot as plt
from matplotlib import use
import importlib

try:
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True

plot_mapped = True

def plot_mapped_polygon(vertex_points: List[NDArray[np.float64]], inertia_vertex_points: List[NDArray[np.float64]], mapped_vertex_points: List[NDArray[np.float64]]) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    ax = axes[0]
    i = 1
    for vertex in vertex_points:
        num_vertices = vertex.shape[1]
        coord = np.zeros([3, num_vertices + 1])
        coord[:, 0:num_vertices] = vertex
        coord[:, num_vertices] = vertex[:, 0]
        ax.plot(coord[0, :], coord[1, :], label='Polygon' + str(i), linewidth=3.0, markersize=12)
        i = i + 1

    ax.title.set_text('original polygon')

    ax = axes[1]
    i = 1
    for vertex in inertia_vertex_points:
        num_vertices = vertex.shape[1]
        coord = np.zeros([3, num_vertices + 1])
        coord[:, 0:num_vertices] = vertex
        coord[:, num_vertices] = vertex[:, 0]
        ax.plot(coord[0, :], coord[1, :], label='Polygon' + str(i), linewidth=3.0, markersize=12)
        i = i + 1

    ax.title.set_text('inertia polygon')


    ax = axes[2]
    i = 1
    for vertex in mapped_vertex_points:
        num_vertices = vertex.shape[1]
        coord = np.zeros([3, num_vertices + 1])
        coord[:, 0:num_vertices] = vertex
        coord[:, num_vertices] = vertex[:, 0]
        ax.plot(coord[0, :], coord[1, :], label='Polygon' + str(i), linewidth=3.0, markersize=12)
        i = i + 1

    ax.title.set_text('mapped polygon')

    plt.show()

class TestPolygonUtilities(unittest.TestCase):

    def test_navem_polygon_on_concave(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        num_vertices = 4
        vertex_points = np.zeros([3, num_vertices])
        vertex_points[0, :] = [0.0, 0.5, 0.5, 0.35]
        vertex_points[1, :] = [0.0, 0.0, 0.5, 0.15]

        area = geometry_utilities.polygon_area(vertex_points)
        centroid = geometry_utilities.polygon_centroid(vertex_points, area)
        diameter = geometry_utilities.polygon_diameter(vertex_points)
        list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(vertex_points)

        polygon = NAVEMPolygon(geometry_utilities, vertex_points, diameter, centroid, list_triangles)
        inertia_mapped_points, _ = polygon.map_inertia_inv(vertex_points)
        _, mapped_vertices, _, _= polygon.map_f_inv(vertex_points, 0)

        if plot_mapped:
            plot_mapped_polygon( [vertex_points],
                                 [inertia_mapped_points],
                                 [mapped_vertices])

    def test_polygon_mapping_on_parallelograms(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        # First polygon
        num_vertices = 4
        vertex_points = np.zeros([3, num_vertices])
        vertex_points[0, :] = [1.4, 1.9, 2.4, 1.9]
        vertex_points[1, :] = [0.035, 0.025, 0.035, 0.045]


        area = geometry_utilities.polygon_area(vertex_points)
        centroid = geometry_utilities.polygon_centroid(vertex_points, area)
        diameter = geometry_utilities.polygon_diameter(vertex_points)
        list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(vertex_points)

        polygon = NAVEMPolygon(geometry_utilities, vertex_points, diameter, centroid, list_triangles)

        inertia_mapped_points, _ = polygon.map_inertia_inv(vertex_points)
        _, mapped_vertices, _, _ = polygon.map_f_inv(vertex_points, 0)

        # Second polygon
        other_vertex_points = np.zeros([3, num_vertices])
        other_vertex_points[0, :] = [0.0, 1.0, 1.2, 0.2]
        other_vertex_points[1, :] = [0.02, 0.02, 0.05, 0.05]

        other_area = geometry_utilities.polygon_area(other_vertex_points)
        other_centroid = geometry_utilities.polygon_centroid(other_vertex_points, other_area)
        other_diameter = geometry_utilities.polygon_diameter(other_vertex_points)
        other_list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(other_vertex_points)

        other_polygon = NAVEMPolygon(geometry_utilities, other_vertex_points, other_diameter, other_centroid, other_list_triangles)

        other_inertia_mapped_points, _ = other_polygon.map_inertia_inv(other_vertex_points)
        _, other_mapped_vertices, _, _ = other_polygon.map_f_inv(other_vertex_points, 0)

        if plot_mapped:
            plot_mapped_polygon( [vertex_points, other_vertex_points],
                                 [inertia_mapped_points, other_inertia_mapped_points],
                                 [mapped_vertices, other_mapped_vertices])

        self.assertLess(np.linalg.norm(mapped_vertices - other_mapped_vertices), 1.0e-12)


    def test_polygon_mapping_on_parallelograms_2(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        num_vertices = 4

        # First polygon
        vertex_points = np.zeros([3, num_vertices])
        vertex_points[0, :] = [-3.2931564766475002e-01, 8.2904083598123200e-10, 8.2904083598123200e-10, -3.2881217384530997e-01]
        vertex_points[1, :] = [-3.2683371802085198e-01, -3.2970894474863799e-01, -8.3629081437663899e-10, -8.3629081437663899e-10]

        area = geometry_utilities.polygon_area(vertex_points)
        centroid = geometry_utilities.polygon_centroid(vertex_points, area)
        diameter = geometry_utilities.polygon_diameter(vertex_points)
        list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(vertex_points)

        polygon = NAVEMPolygon(geometry_utilities, vertex_points, diameter, centroid, list_triangles)

        inertia_mapped_points, _ = polygon.map_inertia_inv(vertex_points)
        _, mapped_vertices, _, _ = polygon.map_f_inv(vertex_points, 0)

        # Second Polygon
        other_vertex_points = np.zeros([3, num_vertices])
        other_vertex_points[0, :] = [-3.29e-01, 8.29e-10, 8.29e-10, -3.29e-01]
        other_vertex_points[1, :] = [-3.29e-01, -3.29e-01, 8.29e-10, 8.29e-10]

        other_area = geometry_utilities.polygon_area(other_vertex_points)
        other_centroid = geometry_utilities.polygon_centroid(other_vertex_points, other_area)
        other_diameter = geometry_utilities.polygon_diameter(other_vertex_points)
        other_list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(other_vertex_points)

        other_polygon = NAVEMPolygon(geometry_utilities, other_vertex_points, other_diameter, other_centroid, other_list_triangles)

        other_inertia_mapped_points, _ = other_polygon.map_inertia_inv(other_vertex_points)
        _, other_mapped_vertices, _, _ = other_polygon.map_f_inv(other_vertex_points, 0)

        if plot_mapped:
            plot_mapped_polygon([vertex_points, other_vertex_points],
                                [inertia_mapped_points, other_inertia_mapped_points],
                                [mapped_vertices, other_mapped_vertices])

    def test_polygon_mapping_on_concave(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        # Polygon mapped as standard
        num_vertices = 4
        vertex_points = np.zeros([3, num_vertices])
        vertex_points[0, :] = [9.91463, 1.93238, -12.832, 1]
        vertex_points[1, :] = [-14.0879, 11.3091, -9.16793, 0]

        # vertex_points[0, :] = [1., 0.13868399, 0., 0.30197903]
        # vertex_points[1, :] = [1., 0.72279196, 0.90146038, 0.]

        area = geometry_utilities.polygon_area(vertex_points)
        centroid = geometry_utilities.polygon_centroid(vertex_points, area)
        list_triangles = geometry_utilities.polygon_triangulation_by_ear_clipping(vertex_points)
        diameter = geometry_utilities.polygon_diameter(vertex_points)

        polygon = NAVEMPolygon(geometry_utilities, vertex_points, diameter, centroid, list_triangles)

        inertia_mapped_points, _ = polygon.map_inertia_inv(vertex_points)
        _, mapped_vertices, _, _ = polygon.map_f_inv(vertex_points, 3)


        # Polygon mapped by applying the maps to its kernel
        polygon_internal_angles = compute_polygon_interior_angles(vertex_points)
        kernel = compute_polygon_kernel(geometry_utilities, vertex_points, polygon_internal_angles)
        kernel_area = geometry_utilities.polygon_area(kernel)

        if kernel_area < geometry_utilities.tolerance2_d() or kernel_area < area * 0.2:
            raise ValueError("Not valid polygon")

        kernel_centroid = geometry_utilities.polygon_centroid(kernel, kernel_area)
        other_centroid = kernel_centroid
        other_diameter = geometry_utilities.polygon_diameter(kernel)
        other_list_triangles =  geometry_utilities.polygon_triangulation_by_ear_clipping(kernel)

        other_polygon = NAVEMPolygon(geometry_utilities, vertex_points, other_diameter, other_centroid, other_list_triangles,True, kernel)

        other_inertia_mapped_points, _ = other_polygon.map_inertia_inv(vertex_points)
        _, other_mapped_vertices, _, _ = other_polygon.map_f_inv(vertex_points, 3)

        if plot_mapped:
            plot_mapped_polygon([vertex_points, vertex_points],
                                [inertia_mapped_points, other_inertia_mapped_points],
                                [mapped_vertices, other_mapped_vertices])


if __name__ == '__main__':

    unittest.main()
