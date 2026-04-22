import unittest

from src.NAVEM.LaplaceSolver import LaplaceSolver, LaplaceProblem
from src.NAVEM.FunctionUtilities import *
from src.GeDiM.geometry.geometry_utilities import *

class TestLaplaceSolver(unittest.TestCase):

    def test_laplace_solver(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-12
        geometry_utilities_config.tolerance2_d = 1.0e-12
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        num_vertices = 5
        domain_vertices = np.zeros([3, num_vertices])
        domain_vertices[0, :] = [1.0, 1.0, -1.0, -1.0, 1.0]
        domain_vertices[1, :] = [0.0, 1.0, 1.0, -1.0, -1.0]
        internal_point = np.zeros([3, 1])
        list_pole_vertices = [0]
        iterations = [50]
        domain_scale = 2.0

        def boundary_conditions(v: int, points: NDArray[np.float64]) -> NDArray[np.float64]:

            if v == 4:
                return 1.0 + points[1, :]
            elif v == 0:
                return 1.0 - points[1, :]
            else:
                return 0.0 + 0.0 * points[1, :]

        problem = LaplaceProblem(domain_vertices, domain_scale, internal_point, boundary_conditions)

        solver = LaplaceSolver(geometry_utilities, problem, iterations, list_pole_vertices)

        num_rat_points = 30
        err_max_l2, err_max_h1 = evaluate_accuracy_on_domain_boundary(geometry_utilities,
                                                                      10 * num_rat_points * solver.problem.num_vertices,
                                                                      solver)

        self.assertLess(err_max_l2, b=1.0e-7)
        self.assertLess(err_max_h1, b=1.0e-7)

        print('Max error L2 {:<.16e} - Max Error H1 {:<.16e} on test points on the boundary.'
              .format(err_max_l2, err_max_h1))

        plot_points_distributions(solver, solver.flat_poles, solver.boundary_points)

        plot_function_and_tangent_derivatives_on_edges(geometry_utilities,
                                                       10 * num_rat_points *
                                                       solver.problem.num_vertices,
                                                       solver)


    def test_laplace_solver_generic(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-12
        geometry_utilities_config.tolerance2_d = 1.0e-12
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        num_vertices = 4
        domain_vertices = np.zeros([3, num_vertices])
        domain_vertices[0, :] = [0.0, 1.0, 1.0, 0.7]
        domain_vertices[1, :] = [0.0, 0.0, 1.0, 0.3]
        domain_scale = 0.1
        vem_id = 1
        iterations = np.arange(3, 120 + 1).tolist()
        list_pole_vertices = np.arange(0, num_vertices).tolist()

        def boundary_conditions(v: int, points: NDArray[np.float64]) -> NDArray[np.float64]:

            prev_id = vem_id - 1 if vem_id > 0 else num_vertices - 1
            next_id = (vem_id + 1) % num_vertices

            if v == vem_id:
                t = (domain_vertices[:, next_id]
                     - domain_vertices[:, vem_id])
                norm_squared_t = t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
                alpha = ((points - domain_vertices[:, vem_id]).T @ t) / norm_squared_t
                return 1.0 - alpha
            elif v == prev_id:
                t = (domain_vertices[:, prev_id]
                     - domain_vertices[:, vem_id])
                norm_squared_t = t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
                alpha = ((points - domain_vertices[:, vem_id]).T @ t) / norm_squared_t
                return 1.0 - alpha
            else:
                return 0.0 * points[1, :]


        internal_angles = compute_polygon_interior_angles(domain_vertices)
        kernel = compute_polygon_kernel(geometry_utilities, domain_vertices, internal_angles)
        area = geometry_utilities.polygon_area(kernel)
        internal_point = geometry_utilities.polygon_centroid(kernel, area)

        problem = LaplaceProblem(domain_vertices, domain_scale, internal_point, boundary_conditions)

        solver = LaplaceSolver(geometry_utilities, problem, iterations, list_pole_vertices)

        num_rat_points = 30
        err_max_l2, err_max_h1 = evaluate_accuracy_on_domain_boundary(geometry_utilities,
                                                                      10 * num_rat_points * solver.problem.num_vertices,
                                                                      solver)

        # self.assertLess(err_max_l2, b=1.0e-7)
        # self.assertLess(err_max_h1, b=1.0e-7)

        print('Max error L2 {:<.16e} - Max Error H1 {:<.16e} on test points on the boundary.'
              .format(err_max_l2, err_max_h1))

        plot_points_distributions(solver, solver.flat_poles, solver.boundary_points)

        plot_function_and_tangent_derivatives_on_edges(geometry_utilities,
                                                       10 * num_rat_points *
                                                       solver.problem.num_vertices,
                                                       solver)


    def test_laplace_solver_concave(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-12
        geometry_utilities_config.tolerance2_d = 1.0e-12
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        num_vertices = 5
        domain_vertices = np.zeros([3, num_vertices])
        domain_vertices[0, :] = [1.0, 2.0, -1.0, -1.0, 2.0]
        domain_vertices[1, :] = [0.0, 1.0, 1.0, -1.0, -1.0]
        internal_point = np.zeros([3, 1])
        list_poles_vertices = [4, 0, 1]
        iterations = [60]
        domain_scale = 3.0

        def boundary_conditions(v: int, points: NDArray[np.float64]) -> NDArray[np.float64]:

            if v == 4:
                return 2.0 - points[0, :]
            elif v == 0:
                return 2.0 - points[0, :]
            else:
                return 0.0 * points[0, :]

        problem = LaplaceProblem(domain_vertices, domain_scale, internal_point, boundary_conditions)

        solver = LaplaceSolver(geometry_utilities, problem, iterations, list_poles_vertices)

        num_rat_points = 30
        err_max_l2, err_max_h1 = evaluate_accuracy_on_domain_boundary(geometry_utilities,
                                                                      10 * num_rat_points * solver.problem.num_vertices,
                                                                      solver)

        self.assertLess(err_max_l2, b=1.0e-5)
        self.assertLess(err_max_h1, b=1.0e-5)

        print('Max error L2 {:<.16e} - Max Error H1 {:<.16e} on test points on the boundary.'
              .format(err_max_l2, err_max_h1))

        plot_points_distributions(solver, solver.flat_poles, solver.boundary_points)

        plot_function_and_tangent_derivatives_on_edges(geometry_utilities,
                                                       10 * num_rat_points *
                                                       solver.problem.num_vertices,
                                                       solver)


    def test_laplace_solver_spike(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-5
        geometry_utilities_config.tolerance2_d = 1.0e-8
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)


        num_vertices = 8
        mag = 0.3
        domain_vertices = np.zeros([3, num_vertices])
        domain_vertices[0, :] = [(1.0 - mag), 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        domain_vertices[1, :] = [0.5, 0.7, 1.0, 1.0, 0.5, 0.0, 0.0, 0.3]
        list_poles_vertices = np.arange(0, 8).tolist()
        internal_point = np.zeros([3, 1])
        domain_scale = 1.0
        iterations = [60]

        def boundary_conditions(v: int, points: NDArray[np.float64]) -> NDArray[np.float64]:

            if v == 0 or v == 7:
                return 1.0 - (1.0/mag) * (points[0, :] - (1.0 - mag))
            else:
                return 0.0 * points[0, :]

        problem = LaplaceProblem(domain_vertices, domain_scale, internal_point, boundary_conditions)

        solver = LaplaceSolver(geometry_utilities, problem, iterations, list_poles_vertices)

        num_rat_points = 30
        err_max_l2, err_max_h1 = evaluate_accuracy_on_domain_boundary(geometry_utilities,
                                                                      10 * num_rat_points * solver.problem.num_vertices,
                                                                      solver)

        self.assertLess(err_max_l2, b=1.0e-2)
        self.assertLess(err_max_h1, b=1.0e-2)

        print('Max error L2 {:<.16e} - Max Error H1 {:<.16e} on test points on the boundary.'
              .format(err_max_l2, err_max_h1))

        plot_points_distributions(solver, solver.flat_poles, solver.boundary_points)

        plot_function_and_tangent_derivatives_on_edges(geometry_utilities,
                                                       10 * num_rat_points *
                                                       solver.problem.num_vertices,
                                                       solver)

    # def test_map_hanging_functions_1(self):
    #
    #     geometry_utilities = GeometryUtilities()
    #     num_rat_points = 30
    #     harm_deg = 20
    #
    #     hanging_functions = HangingFunction(geometry_utilities, harm_deg, num_rat_points, 'real', 'real')
    #
    #     # Export hanging function on mapped polygon
    #     vertices = np.zeros([3, 5])
    #     vertices[0, :] = [1.0, 1.8, 2.0, 1.0, 0.5]
    #     vertices[1, :] = [2.0, 3.6, 4.0, 4.0, 2.5]
    #
    #     vertices = np.array([[0.0, 0.25,  0.5, 0.5, 0.0],
    #                         [0.25,  0.2875, 0.25,  0.5, 0.5],
    #                         [0.0, 0.0, 0.0, 0.0,  0.]])
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       hanging_functions.num_vertices,
    #                                                                                       hanging_functions,
    #                                                                                       angles,
    #                                                                                       vertex_distance)
    #
    # def test_map_hanging_functions_2(self):
    #
    #     geometry_utilities = GeometryUtilities()
    #     num_rat_points = 30
    #     harm_deg = 20
    #
    #     hanging_functions = HangingFunction(geometry_utilities, harm_deg, num_rat_points, 'real', 'real')
    #
    #     tri_hang_generator = TriangleDFNGenerator(geometry_utilities)
    #     train_mesh, _ = (
    #         tri_hang_generator.
    #         generate_training_triangle_with_hanging_from_cartesian(1, False,
    #                                                                False, False))
    #
    #     vertices = train_mesh.get_cell_2_vertices_coordinate(0)
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       hanging_functions.num_vertices,
    #                                                                                       hanging_functions,
    #                                                                                       angles,
    #                                                                                       vertex_distance)
    #
    # def test_map_hanging_functions_3(self):
    #
    #     geometry_utilities = GeometryUtilities()
    #     num_rat_points = 30
    #     harm_deg = 20
    #
    #     hanging_functions = HangingFunction(geometry_utilities, harm_deg, num_rat_points, 'real', 'real', 1.5*np.pi)
    #
    #     domain_vertices = np.array([[0.0, 1.0,  1.0, 0.0],
    #                                 [0.0,  0.0, 1.0, 1.0],
    #                                 [0.0, 0.0, 0.0, 0.0]])
    #     rectangle_origin = domain_vertices[:, 0]
    #     rectangle_base_tangent = domain_vertices[:, 1] - domain_vertices[:, 0]
    #     rectangle_height_tangent = domain_vertices[:, 3] - domain_vertices[:, 0]
    #
    #     mesh = mesh_generators.create_structured_concave_mesh(rectangle_origin, rectangle_base_tangent,
    #                                                           rectangle_height_tangent,
    #                                                           1, 1)
    #
    #     vertices = mesh.get_cell_2_vertices_coordinate(0)
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       hanging_functions.num_vertices,
    #                                                                                       hanging_functions,
    #                                                                                       angles,
    #                                                                                       vertex_distance)
    #
    #     vertices = mesh.get_cell_2_vertices_coordinate(1)
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       hanging_functions.num_vertices,
    #                                                                                       hanging_functions,
    #                                                                                       angles,
    #                                                                                       vertex_distance)
    #
    #     vertices = mesh.get_cell_2_vertices_coordinate(2)
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       hanging_functions.num_vertices,
    #                                                                                       hanging_functions,
    #                                                                                       angles,
    #                                                                                       vertex_distance)
    #
    # def test_map_solver_functions_4(self):
    #
    #     geometry_utilities = GeometryUtilities()
    #     num_rat_points = 30
    #
    #     solver = LaplaceSolver('polygon_concave', num_iterations=50, function_type='total')
    #
    #     domain_vertices = np.array([[0.0, 1.0,  1.0, 0.0],
    #                                 [0.0,  0.0, 1.0, 1.0],
    #                                 [0.0, 0.0, 0.0, 0.0]])
    #
    #     rectangle_origin = domain_vertices[:, 0]
    #     rectangle_base_tangent = domain_vertices[:, 1] - domain_vertices[:, 0]
    #     rectangle_height_tangent = domain_vertices[:, 3] - domain_vertices[:, 0]
    #
    #     mesh = mesh_generators.create_structured_concave_mesh(rectangle_origin, rectangle_base_tangent,
    #                                                           rectangle_height_tangent,
    #                                                           1, 1)
    #
    #     vertices = mesh.get_cell_2_vertices_coordinate(0)
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       solver.num_vertices,
    #                                                                                       solver,
    #                                                                                       angles,
    #                                                                                       vertex_distance)
    #
    #     vertices = mesh.get_cell_2_vertices_coordinate(1)
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       solver.num_vertices,
    #                                                                                       solver,
    #                                                                                       angles,
    #                                                                                       vertex_distance)
    #
    #     vertices = mesh.get_cell_2_vertices_coordinate(2)
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       solver.num_vertices,
    #                                                                                       solver,
    #                                                                                       angles,
    #                                                                                       vertex_distance)
    #
    # def test_map_solver_functions_2(self):
    #
    #     geometry_utilities = GeometryUtilities()
    #     num_rat_points = 30
    #     solver = LaplaceSolver('polygon_hanging')
    #
    #     tri_hang_generator = TriangleDFNGenerator(geometry_utilities)
    #     train_mesh, _ = (
    #         tri_hang_generator.
    #         generate_training_triangle_with_hanging_from_cartesian(1, False,
    #                                                                False, False))
    #
    #     vertices = train_mesh.get_cell_2_vertices_coordinate(0)
    #     HangingFunctionUtilities.plot_function_and_tangent_derivatives_on_mapped_elements(geometry_utilities, vertices,
    #                                                                                       10 * num_rat_points *
    #                                                                                       solver.num_vertices,
    #                                                                                       solver)
    #
    # def test_map_solver_functions_on_polygons(self):
    #     geometry_utilities = GeometryUtilities()
    #     solver = LaplaceSolver('polygon_hanging')
    #
    #     pol_generator = polygon_generators.PolygonGenerator(geometry_utilities)
    #     train_mesh = pol_generator.generate_polygons(2, 0, 1,
    #                                                  5, False)
    #
    #     vertices = train_mesh.get_cell_2_vertices_coordinate(1)
    #     diameter = geometry_utilities.compute_polygon_diameter(vertices)
    #     area = geometry_utilities.compute_polygon_area(vertices)
    #     centroid = geometry_utilities.compute_polygon_centroid(vertices, area)
    #     polygon = NAVEMPolygon(geometry_utilities, vertices, diameter, centroid, centroid)
    #
    #     polygon_vertices = polygon.map_f_inv(polygon.vertices, 4)[0]
    #
    #     ##################################################################
    #
    #     fig = plt.figure(figsize=plt.figaspect(0.5))
    #     plt.rc('legend', fontsize=30)
    #     ax = fig.add_subplot()
    #
    #     #################################################################
    #
    #     coord_x = 1
    #     coord_y = 0
    #
    #     vertex_distance = []
    #     for v in range(vertices.shape[1]):
    #         vertex_distance.append(
    #             geometry_utilities.compute_polygon_max_distance(v, vertices))
    #
    #     internal_angles = geometry_utilities.compute_polygon_interior_angles(vertices)
    #
    #     for v in [3, 4, 0]:
    #
    #         translation_1, translation_2, B, B_inv = HangingFunctionUtilities.compute_map_f(geometry_utilities, v,
    #                                                                                         polygon_vertices,
    #                                                                                         solver.polygon_vertices,
    #                                                                                         internal_angles,
    #                                                                                         vertex_distance)
    #
    #         mapped_reference_vertices, _ = HangingFunctionUtilities.map_f(translation_1, translation_2, B, B_inv, solver.polygon_vertices)
    #
    #         ##################################################################
    #
    #         coord = np.zeros([3, solver.num_vertices + 1])
    #         coord[:, 0:solver.num_vertices] = mapped_reference_vertices
    #         coord[:, solver.num_vertices] = mapped_reference_vertices[:, 0]
    #         if v == 4:
    #             ax.plot(coord[coord_x, :], coord[coord_y, :], label='$\\Omega^{5}_{\\Phi,5,E}$',
    #                     linewidth=3.0, marker='o', markersize=10)
    #         elif v == 0:
    #             ax.plot(coord[coord_x, :], coord[coord_y, :], label='$\\Omega^{1}_{\\Phi,5,E}$',
    #                     linewidth=3.0, marker='o', markersize=10)
    #         elif v == 3:
    #             ax.plot(coord[coord_x, :], coord[coord_y, :], label='$\\Omega^{4}_{\\Phi,5,E}$',
    #                     linewidth=3.0, marker='o', markersize=10)
    #
    #         #################################################################
    #
    #     ##################################################################
    #
    #     coord = np.zeros([3, polygon_vertices.shape[1] + 1])
    #     coord[:, 0:polygon_vertices.shape[1]] = polygon_vertices
    #     coord[:, polygon_vertices.shape[1]] = polygon_vertices[:, 0]
    #     ax.plot(coord[coord_x, :], coord[coord_y, :], label='$E$',linewidth=3.0, marker='o', markersize = 10)
    #
    #     # Add numbers at each vertex and intersection of the hexagram
    #     list_vertices = []
    #     for i in range(polygon_vertices.shape[1]):
    #         list_vertices.append((polygon_vertices[coord_x, i], polygon_vertices[coord_y, i]))
    #
    #     for i, (x, y) in enumerate(list_vertices, start=1):
    #         if i == 1 or i == 5:
    #             plt.annotate(str(i), (x + 1.0e-01, y + 1.0e-01), ha='center', va='center', size=30)
    #         elif i == 2:
    #             plt.annotate(str(i), (x - 1.0e-01, y + 1.0e-01), ha='center', va='center', size=30)
    #         else:
    #             plt.annotate(str(i), (x - 1.0e-01, y - 1.0e-01), ha='center', va='center', size=30)
    #
    #     #################################################################
    #
    #     plt.legend(bbox_to_anchor=(0.1, 1.02, 0.8, 0.2), loc="lower left",
    #                mode="expand", borderaxespad=0, ncol=4)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     ax.set_aspect('equal', 'box')
    #
    #     if coord_x == 0:
    #         plt.xlabel('${x_1}$', fontsize=30)
    #         plt.ylabel('${x_2}$', fontsize=30)
    #     else:
    #         plt.xlabel('${x_2}$', fontsize=30)
    #         plt.ylabel('${x_1}$', fontsize=30)
    #     plt.show()
    #
    #     ##################################################################
    #
    # def test_divergence_free_polynomials(self):
    #
    #     degree = 3
    #     df_monomials = DivergenceFreePolynomial(degree)
    #     monomials = Monomials2D(degree)
    #
    #     num_points = 2
    #     points = np.zeros([3, 2])
    #     points[:, 0] = [0.0, 0.1, 0.0]
    #     points[:, 1] = [0.5, -0.6, 0.0]
    #
    #     centroid = np.zeros([3, 1])
    #     centroid[0, 0] = -0.2
    #
    #     df_vander, _ = df_monomials.vander(points, 0.3, centroid)
    #     vander = monomials.vandermonde(points, 0.3, centroid)
    #
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 0] - np.zeros([num_points])), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 0] - vander[:, 0]), 1.0e-12)
    #
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 1] - np.zeros([num_points])), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 1] - vander[:, 0]), 1.0e-12)
    #
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 2] - np.zeros([num_points])), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 2] - vander[:, 1]), 1.0e-12)
    #
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 3] - vander[:, 1]), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 3] + vander[:, 2]), 1.0e-12)
    #
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 4] - vander[:, 2]), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 4] - np.zeros([num_points])), 1.0e-12)
    #
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 5] - np.zeros([num_points])), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 5] - vander[:, 3]), 1.0e-12)
    #
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 6] - 0.5 * vander[:, 3]), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 6] + vander[:, 4]), 1.0e-12)
    #
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 7] - vander[:, 4]), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 7] + 0.5 * vander[:, 5]), 1.0e-12)
    #
    #     self.assertLess(np.linalg.norm(df_vander[0, :, 8] - vander[:, 5]), 1.0e-12)
    #     self.assertLess(np.linalg.norm(df_vander[1, :, 8] - np.zeros([num_points])), 1.0e-12)


if __name__ == '__main__':

    unittest.main()

