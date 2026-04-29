import unittest
from src.NAVEM.Utilities.points_generator import *
from src.GeDiM.geometry.geometry_utilities import *


try:
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True


class TestNodeDistribution(unittest.TestCase):

    def test_triangle_1(self):

        reference_triangle = np.zeros([3, 3])
        reference_triangle[0, 1] = 1.0
        reference_triangle[1, 2] = 1.0

        m = 3
        xy_points = uniform_grid_over_triangle(m, boundary = False, rescale = False)
        plot_grid_over_polygon(reference_triangle, xy_points)

    def test_triangle_2(self):

        reference_triangle = np.zeros([3, 3])
        reference_triangle[0, 1] = 1.0
        reference_triangle[1, 2] = 1.0

        m = 3
        # Test 2:
        xy_points = uniform_grid_over_triangle(m, boundary = True, rescale = False)
        plot_grid_over_polygon(reference_triangle, xy_points)

    def test_triangle_3(self):
        m = 3

        # Test 3:

        reference_triangle = np.zeros([3, 3])
        reference_triangle[0, 1] = 1.0
        reference_triangle[1, 2] = 1.0

        border_type = BorderType.all_borders
        power = 0.5

        xy_points = concentrating_grid_over_triangle(m, power=power, border_type=border_type)
        plot_grid_over_polygon(reference_triangle, xy_points)

    def test_polygon_1(self):

        m = 3
        # border_type = 0
        # power = 0.75
        border_type = BorderType.no_borders
        power = 1

        polygon = np.array([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_internal_angles = compute_polygon_interior_angles(polygon)
        kernel = compute_polygon_kernel(geometry_utilities, polygon, polygon_internal_angles)
        kernel_area = geometry_utilities.polygon_area(kernel)
        internal_point = geometry_utilities.polygon_centroid(kernel, kernel_area)
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(polygon, internal_point)
        list_triangulation_points = geometry_utilities.extract_triangulation_points_by_internal_point(polygon, internal_point, list_triangles)

        # Test 4:
        xy_points = uniform_grid_over_triangle(m, boundary=False, rescale=True, polygon=polygon.shape[1] > 3)
        nodes = uniform_grid_over_polygon(xy_points, list_triangulation_points, boundary=False)
        plot_grid_over_polygon(polygon, nodes)

        # Test 5:
        xy_points = uniform_grid_over_triangle(m, boundary=True, polygon=polygon.shape[1] > 3)
        nodes = uniform_grid_over_polygon(xy_points, list_triangulation_points, boundary=True)
        plot_grid_over_polygon(polygon, nodes)

        # Test 6:
        xy_points = concentrating_grid_over_triangle(m, power=power, border_type=border_type)
        nodes = concentrating_grid_over_polygon(xy_points, list_triangulation_points, border_type)
        plot_grid_over_polygon(polygon, nodes)

    def test_polygon_2(self):

        m = 3
        border_type = BorderType.no_borders
        power = 1

        polygon = np.array([[-1.2, 0.7, 1.1, -1.1], [-0.6, -1.2, 1.1, 0.7], [0.0, 0.0, 0.0, 0.0]])

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_internal_angles = compute_polygon_interior_angles(polygon)
        kernel = compute_polygon_kernel(geometry_utilities, polygon, polygon_internal_angles)
        kernel_area = geometry_utilities.polygon_area(kernel)
        internal_point = geometry_utilities.polygon_centroid(kernel, kernel_area)
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(polygon, internal_point)
        list_triangulation_points = geometry_utilities.extract_triangulation_points_by_internal_point(polygon, internal_point, list_triangles)

        # Test 4:
        xy_points = uniform_grid_over_triangle(m, boundary=False, rescale=True, polygon=polygon.shape[1] > 3)
        nodes = uniform_grid_over_polygon(xy_points, list_triangulation_points, boundary=False)
        plot_grid_over_polygon(polygon, nodes)

        # Test 5:
        xy_points = uniform_grid_over_triangle(m, boundary=True, polygon=polygon.shape[1] > 3)
        nodes = uniform_grid_over_polygon(xy_points, list_triangulation_points, boundary=True)
        plot_grid_over_polygon(polygon, nodes)

        # Test 6:
        xy_points = concentrating_grid_over_triangle(m, power=power, border_type=border_type)
        nodes = concentrating_grid_over_polygon(xy_points, list_triangulation_points, border_type)
        plot_grid_over_polygon(polygon, nodes)

    def test_polygon_3(self):

        m = 3
        border_type = BorderType.no_borders
        power = 1


        polygon = np.array([[3, 5, 4, 3, 1.5, 1], [0, 1.5, 4, 3.5, 4, 2], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_internal_angles = compute_polygon_interior_angles(polygon)
        kernel = compute_polygon_kernel(geometry_utilities, polygon, polygon_internal_angles)
        kernel_area = geometry_utilities.polygon_area(kernel)
        internal_point = geometry_utilities.polygon_centroid(kernel, kernel_area)
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(polygon, internal_point)
        list_triangulation_points = geometry_utilities.extract_triangulation_points_by_internal_point(polygon,
                                                                                                      internal_point,
                                                                                                      list_triangles)

        # Test 4:
        xy_points = uniform_grid_over_triangle(m, boundary=False, rescale=True, polygon=polygon.shape[1] > 3)
        nodes = uniform_grid_over_polygon(xy_points, list_triangulation_points, boundary=False)
        plot_grid_over_polygon(polygon, nodes)

        # Test 5:
        xy_points = uniform_grid_over_triangle(m, boundary=True, polygon=polygon.shape[1] > 3)
        nodes = uniform_grid_over_polygon(xy_points, list_triangulation_points, boundary=True)
        plot_grid_over_polygon(polygon, nodes)

        # Test 6:
        xy_points = concentrating_grid_over_triangle(m, power=power, border_type=border_type)
        nodes = concentrating_grid_over_polygon(xy_points, list_triangulation_points, border_type)
        plot_grid_over_polygon(polygon, nodes)

    def test_polygon_4(self):

        m = 7
        border_type = BorderType.no_borders
        power = 0.75

        polygon = np.array(
            [[-0.65, -0.75, -0.05, 0.6, 0.85, 0.1], [0.45, -0.35, -1.0, -0.7, 0.45, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        polygon_internal_angles = compute_polygon_interior_angles(polygon)
        kernel = compute_polygon_kernel(geometry_utilities, polygon, polygon_internal_angles)
        kernel_area = geometry_utilities.polygon_area(kernel)
        internal_point = geometry_utilities.polygon_centroid(kernel, kernel_area)
        list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(polygon, internal_point)
        list_triangulation_points = geometry_utilities.extract_triangulation_points_by_internal_point(polygon,
                                                                                                      internal_point,
                                                                                                      list_triangles)

        # Test 4:
        xy_points = uniform_grid_over_triangle(m, boundary=True, rescale=True, polygon=polygon.shape[1] > 3)
        nodes = uniform_grid_over_polygon(xy_points, list_triangulation_points, boundary=True)
        plot_grid_over_polygon(polygon, nodes)

        # Test 5:
        xy_points = uniform_grid_over_triangle(m, boundary=True, polygon=polygon.shape[1] > 3)
        nodes = uniform_grid_over_polygon(xy_points, list_triangulation_points, boundary=True)
        plot_grid_over_polygon(polygon, nodes)

        # Test 6:
        xy_points = concentrating_grid_over_triangle(m, power=power, border_type=border_type)
        nodes = concentrating_grid_over_polygon(xy_points, list_triangulation_points, border_type)
        plot_grid_over_polygon(polygon, nodes)



if __name__ == '__main__':
    unittest.main()
