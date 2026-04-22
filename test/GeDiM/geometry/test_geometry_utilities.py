import unittest
from src.GeDiM.geometry.geometry_utilities import *
import numpy as np
from pypolydim import gedim
import matplotlib.pyplot as plt

def plot_polygon_with_external_bisectors(polygon_vertices: np.ndarray, external_bisectors: np.ndarray, scale=0.5):

    num_vertices = polygon_vertices.shape[1]

    polygon_vertices_closed = np.zeros([3, num_vertices + 1])
    polygon_vertices_closed[:, :-1] = polygon_vertices
    polygon_vertices_closed[:, -1] = polygon_vertices[:, 0]

    plt.figure(figsize=(6, 6))

    plt.plot(polygon_vertices_closed[0, :], polygon_vertices_closed[1, :], 'k-', lw=2)
    plt.scatter(polygon_vertices[0, :], polygon_vertices[1, :], color='black')

    for i in range(num_vertices):
        p = polygon_vertices[:, i]
        b = external_bisectors[:, i]
        q = polygon_vertices[:, i] + scale * b
        plt.plot([p[0], q[0]], [p[1], q[1]], 'r--')

    plt.axis('equal')
    plt.grid(True)
    plt.title("Polygon and external bisectors")
    plt.show()

class TestGeometryUtilities(unittest.TestCase):

    def test_compute_polygon_interior_angles_equilateral_triangle(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 3])
        polygon_vertices[0, :] = [0.0, 1.0, 0.5]
        polygon_vertices[1, :] = [0.0, 0.0, np.sqrt(3)/2]

        interior_angles = compute_polygon_interior_angles(polygon_vertices)

        self.assertLess(abs(interior_angles[0] - np.pi / 3.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[1] - np.pi / 3.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[2] - np.pi / 3.0), geometry_utilities.tolerance1_d())

    def test_compute_polygon_interior_angles_square(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 4])
        polygon_vertices[0, :] = [0.0, 1.0, 1.0, 0.0]
        polygon_vertices[1, :] = [0.0, 0.0, 1.0, 1.0]

        interior_angles = compute_polygon_interior_angles(polygon_vertices)

        self.assertLess(abs(interior_angles[0] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[1] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[2] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[3] - np.pi / 2.0), geometry_utilities.tolerance1_d())

    def test_compute_polygon_interior_angles_convex_polygon(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 5])
        polygon_vertices[0, :] = [0.0, 2.0, 3.0, 1.5, 0.0]
        polygon_vertices[1, :] = [0.0, 0.0, 1.0, 3.0, 1.0]

        interior_angles = compute_polygon_interior_angles(polygon_vertices)

        self.assertLess(abs(sum(interior_angles) - np.pi * 3), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[0] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[1] - 135.0 * np.pi / 180.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[2] - 1.7126933813990606), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[3] - 1.2870022175865687), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[4] - 2.498091544796509), geometry_utilities.tolerance1_d())

    def test_compute_polygon_interior_angles_aligned_edges(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 5])
        polygon_vertices[0, :] = [0.0, 1.0, 2.0, 2.0, 0.0]
        polygon_vertices[1, :] = [0.0, 0.0, 0.0, 1.0, 1.0]

        interior_angles = compute_polygon_interior_angles(polygon_vertices)

        self.assertLess(abs(sum(interior_angles) - np.pi * 3), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[0] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[1] - np.pi), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[2] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[3] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[4] - np.pi / 2.0), geometry_utilities.tolerance1_d())

    def test_compute_polygon_interior_angles_concave_polygon(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 5])
        polygon_vertices[0, :] = [0.0, 2.0, 2.0, 1.0, 0.0]
        polygon_vertices[1, :] = [0.0, 0.0, 2.0, 1.0, 2.0]

        interior_angles = compute_polygon_interior_angles(polygon_vertices)

        self.assertLess(abs(sum(interior_angles) - np.pi * 3), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[0] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[1] - np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[2] - np.pi / 4.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[3] - 3.0 * np.pi / 2.0), geometry_utilities.tolerance1_d())
        self.assertLess(abs(interior_angles[4] - np.pi / 4.0), geometry_utilities.tolerance1_d())
        
    def test_compute_polygon_external_bisectors_equilateral_triangle(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 3])
        polygon_vertices[0, :] = [0.0, 1.0, 0.5]
        polygon_vertices[1, :] = [0.0, 0.0, np.sqrt(3)/2]

        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        external_bisectors = compute_polygon_external_bisectors(geometry_utilities, polygon_edge_normals)

        # plot_polygon_with_external_bisectors(polygon_vertices, external_bisectors)

        self.assertLess(np.linalg.norm(external_bisectors[:, 0] - np.array([-np.sqrt(3.0)/2.0, -0.5, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 1] - np.array([np.sqrt(3.0)/2.0, -0.5, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 2] - np.array([0.0, 1.0, 0.0])), geometry_utilities.tolerance1_d())


    def test_compute_polygon_external_bisectors_square(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 4])
        polygon_vertices[0, :] = [0.0, 1.0, 1.0, 0.0]
        polygon_vertices[1, :] = [0.0, 0.0, 1.0, 1.0]

        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        external_bisectors = compute_polygon_external_bisectors(geometry_utilities, polygon_edge_normals)

        # plot_polygon_with_external_bisectors(polygon_vertices, external_bisectors)

        self.assertLess(np.linalg.norm(external_bisectors[:, 0] - np.array([-np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 1] - np.array([np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 2] - np.array([np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 3] - np.array([-np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())

    def test_compute_polygon_external_bisectors_convex_polygon(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 5])
        polygon_vertices[0, :] = [0.0, 2.0, 3.0, 1.5, 0.0]
        polygon_vertices[1, :] = [0.0, 0.0, 1.0, 3.0, 1.0]

        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        external_bisectors = compute_polygon_external_bisectors(geometry_utilities, polygon_edge_normals)

        # plot_polygon_with_external_bisectors(polygon_vertices, external_bisectors)

        self.assertLess(np.linalg.norm(external_bisectors[:, 0] - np.array([-np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 1] - np.array([3.8268343236508973e-01, -9.2387953251128674e-01, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 2] - np.array([9.9748420881264244e-01, -7.0889020090679267e-02, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 3] - np.array([0.0, 1.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 4] - np.array([-9.4868329805051388e-01, 3.1622776601683794e-01, 0.0])), geometry_utilities.tolerance1_d())

    def test_compute_polygon_external_bisectors_aligned_edges(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 5])
        polygon_vertices[0, :] = [0.0, 1.0, 2.0, 2.0, 0.0]
        polygon_vertices[1, :] = [0.0, 0.0, 0.0, 1.0, 1.0]

        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        external_bisectors = compute_polygon_external_bisectors(geometry_utilities, polygon_edge_normals)

        # plot_polygon_with_external_bisectors(polygon_vertices, external_bisectors)

        self.assertLess(np.linalg.norm(external_bisectors[:, 0] - np.array([-np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 1] - np.array([0.0, -1.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 2] - np.array([np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 3] - np.array([np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 4] - np.array([-np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())

    def test_compute_polygon_external_bisectors_concave_polygon(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_vertices = np.zeros([3, 5])
        polygon_vertices[0, :] = [0.0, 2.0, 2.0, 1.0, 0.0]
        polygon_vertices[1, :] = [0.0, 0.0, 2.0, 1.0, 2.0]

        polygon_edge_normals = geometry_utilities.polygon_edge_normals(polygon_vertices)
        external_bisectors = compute_polygon_external_bisectors(geometry_utilities, polygon_edge_normals)

        # plot_polygon_with_external_bisectors(polygon_vertices, external_bisectors)

        self.assertLess(np.linalg.norm(external_bisectors[:, 0] - np.array([-np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 1] - np.array([np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 2] - np.array([3.8268343236508989e-01, 9.2387953251128674e-01, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 3] - np.array([0.0, 1.0, 0.0])), geometry_utilities.tolerance1_d())
        self.assertLess(np.linalg.norm(external_bisectors[:, 4] - np.array([-3.8268343236508989e-01, 9.2387953251128674e-01, 0.0])), geometry_utilities.tolerance1_d())



if __name__ == '__main__':
    unittest.main()
