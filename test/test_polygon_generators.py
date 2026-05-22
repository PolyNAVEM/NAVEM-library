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
import NAVEM.PCC_2D.NAVEM_PCC_2D
from NAVEM.geometry.geometry_utilities import *
import matplotlib.pyplot as plt
from matplotlib import use
import importlib
from NAVEM.geometry.polygon_generators import PolygonGenerator

try:
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True


plot_generate_elements = False

class TestPolygonGenerators(unittest.TestCase):

    def test_simple_polygon_generators(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        mesh_utilities = gedim.MeshUtilities()

        n_training_polygons = 10
        n_vertices = 7

        polygon_generator = PolygonGenerator(geometry_utilities)
        mesh_data, mesh = polygon_generator.generate_simple_random_polygon(num_polygons=n_training_polygons,
                                                                           num_vertices=n_vertices,
                                                                           plot_generate_elements=plot_generate_elements)

        self.assertTrue(mesh.cell2_d_total_number() == n_training_polygons)

        _ = compute_geometric_properties_mesh_2(geometry_utilities, mesh_utilities, mesh)

    def test_star_shaped_polygon_generators(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        mesh_utilities = gedim.MeshUtilities()

        n_training_polygons = 10
        n_vertices = 6

        polygon_generator = PolygonGenerator(geometry_utilities)
        mesh_data, mesh = polygon_generator.generate_star_shaped_random_polygon(num_polygons=n_training_polygons,
                                                                                num_vertices=n_vertices,
                                                                                plot_generate_elements=plot_generate_elements)

        _ = compute_geometric_properties_mesh_2(geometry_utilities, mesh_utilities, mesh)

        self.assertTrue(mesh.cell2_d_total_number() == n_training_polygons)

    def test_convex_polygon_generators(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        mesh_utilities = gedim.MeshUtilities()

        n_training_polygons = 10
        n_vertices = 8

        polygon_generator = PolygonGenerator(geometry_utilities)
        mesh_data, mesh \
            = polygon_generator.generate_random_convex_polygon(num_polygons=n_training_polygons,
                                                               num_vertices=n_vertices,
                                                               tolerance_hanging_nodes=NAVEM.PCC_2D.NAVEM_PCC_2D.NNCategory.tolerance_hanging_nodes,
                                                               plot_generate_elements=plot_generate_elements)

        _ = compute_geometric_properties_mesh_2(geometry_utilities, mesh_utilities, mesh)

        self.assertTrue(mesh.cell2_d_total_number() == n_training_polygons)

    def test_convex_quadrilateral_generators(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        mesh_utilities = gedim.MeshUtilities()

        n_training_polygons = 1000
        n_vertices = 4

        polygon_generator = PolygonGenerator(geometry_utilities)
        mesh_data, mesh \
            = polygon_generator.generate_random_convex_polygon(num_polygons=n_training_polygons,
                                                               num_vertices=n_vertices,
                                                               tolerance_hanging_nodes=NAVEM.PCC_2D.NAVEM_PCC_2D.NNCategory.tolerance_hanging_nodes,
                                                               plot_generate_elements=plot_generate_elements)

        _ = compute_geometric_properties_mesh_2(geometry_utilities, mesh_utilities, mesh)

        self.assertTrue(mesh.cell2_d_total_number() == n_training_polygons)


    def test_concave_polygon_generators(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        mesh_utilities = gedim.MeshUtilities()

        n_training_polygons = 10
        n_vertices = 7

        polygon_generator = PolygonGenerator(geometry_utilities)
        mesh_data, mesh \
            = polygon_generator.generate_random_concave_polygon(num_polygons=n_training_polygons,
                                                                num_vertices=n_vertices,
                                                                num_reflex_angles=2,
                                                                tolerance_hanging_nodes=NAVEM.PCC_2D.NAVEM_PCC_2D.NNCategory.tolerance_hanging_nodes,
                                                                plot_generate_elements=plot_generate_elements)

        _ = compute_geometric_properties_mesh_2(geometry_utilities, mesh_utilities, mesh)

        self.assertTrue(mesh.cell2_d_total_number() == n_training_polygons)

    def test_concave_quadrilateral_generators(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        mesh_utilities = gedim.MeshUtilities()

        n_training_polygons = 1000
        n_vertices = 4

        polygon_generator = PolygonGenerator(geometry_utilities)
        mesh_data, mesh \
            = polygon_generator.generate_random_concave_polygon(num_polygons=n_training_polygons,
                                                                num_vertices=n_vertices,
                                                                num_reflex_angles=1,
                                                                tolerance_hanging_nodes=NAVEM.PCC_2D.NAVEM_PCC_2D.NNCategory.tolerance_hanging_nodes,
                                                                plot_generate_elements=plot_generate_elements)

        _ = compute_geometric_properties_mesh_2(geometry_utilities, mesh_utilities, mesh)

        self.assertTrue(mesh.cell2_d_total_number() == n_training_polygons)

if __name__ == '__main__':

    unittest.main()
