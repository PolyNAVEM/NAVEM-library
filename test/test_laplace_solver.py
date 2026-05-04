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

from NAVEM.Utilities.LaplaceSolver import LaplaceSolver, LaplaceProblem
from NAVEM.Utilities.FunctionUtilities import *
from NAVEM.geometry.geometry_utilities import *

plot_function = True

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

        def boundary_tangent_derivatives(v: int, vertices: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:

            if v == 4:
                return (np.ones(points.shape[1])) / np.linalg.norm(vertices[:, 0:1] - vertices[:, 4:5])
            elif v == 0:
                return (-np.ones(points.shape[1])) / np.linalg.norm(vertices[:, 0:1] - vertices[:, 1:2])
            else:
                return 0.0 + 0.0 * points[1, :]

        problem = LaplaceProblem(domain_vertices, domain_scale, internal_point, boundary_conditions)

        solver = LaplaceSolver(geometry_utilities, problem, iterations, list_pole_vertices)

        num_rat_points = 30
        err_max_l2, err_max_h1 = evaluate_accuracy_on_domain_boundary(geometry_utilities,
                                                                      10 * num_rat_points * solver.problem.num_vertices,
                                                                      solver,
                                                                      boundary_conditions,
                                                                      boundary_tangent_derivatives=boundary_tangent_derivatives)

        self.assertLess(err_max_l2, b=1.0e-7)
        self.assertLess(err_max_h1, b=1.0e-7)

        if plot_function:
            print('Max error L2 {:<.16e} - Max Error H1 {:<.16e} on test points on the boundary.'
                  .format(err_max_l2, err_max_h1))

            plot_points_boundary_distributions(solver.domain_vertices(), solver.flat_poles, solver.boundary_points)

            plot_function_and_tangent_derivatives_on_edges(geometry_utilities,
                                                           10 * num_rat_points *
                                                           solver.problem.num_vertices,
                                                           solver,
                                                           boundary_conditions,
                                                           boundary_tangent_derivatives=boundary_tangent_derivatives)


    def test_laplace_solver_generic(self):

        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-12
        geometry_utilities_config.tolerance2_d = 1.0e-12
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)

        num_vertices = 4
        domain_vertices = np.zeros([3, num_vertices])
        domain_vertices[0, :] = [0.0, 1.0, 1.0, 0.7]
        domain_vertices[1, :] = [0.0, 0.0, 1.0, 0.3]
        domain_scale = 1.0
        vem_id = 1
        iterations = [70]
        list_pole_vertices = np.arange(0, num_vertices).tolist()

        def boundary_conditions(v: int, points: NDArray[np.float64]) -> NDArray[np.float64]:

            prev_id = vem_id - 1 if vem_id > 0 else num_vertices - 1
            next_id = (vem_id + 1) % num_vertices

            if v == vem_id:
                t = (domain_vertices[:, next_id]
                     - domain_vertices[:, vem_id])
                norm_squared_t = t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
                alpha = ((points - domain_vertices[:, vem_id:vem_id+1]).T @ t) / norm_squared_t
                return 1.0 - alpha
            elif v == prev_id:
                t = (domain_vertices[:, prev_id]
                     - domain_vertices[:, vem_id])
                norm_squared_t = t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
                alpha = ((points - domain_vertices[:, vem_id:vem_id+1]).T @ t) / norm_squared_t
                return 1.0 - alpha
            else:
                return 0.0 * points[1, :]


        def boundary_tangent_derivatives(v: int, vertices: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:

            prev_id = vem_id - 1 if vem_id > 0 else num_vertices - 1
            next_id = (vem_id + 1) % num_vertices

            if v == vem_id:
                return (-np.ones(points.shape[1])) / np.linalg.norm(vertices[:, vem_id:vem_id+1] - vertices[:, prev_id:prev_id+1])
            elif v == prev_id:
                return (np.ones(points.shape[1])) / np.linalg.norm(vertices[:, vem_id:vem_id+1] - vertices[:, next_id:next_id+1])
            else:
                return 0.0 + 0.0 * points[1, :]

        internal_angles = compute_polygon_interior_angles(domain_vertices)
        kernel = compute_polygon_kernel(geometry_utilities, domain_vertices, internal_angles)
        area = geometry_utilities.polygon_area(kernel)
        internal_point = geometry_utilities.polygon_centroid(kernel, area)

        problem = LaplaceProblem(domain_vertices, domain_scale, internal_point, boundary_conditions)

        solver = LaplaceSolver(geometry_utilities, problem, iterations, list_pole_vertices)

        num_rat_points = 30
        err_max_l2, err_max_h1 = evaluate_accuracy_on_domain_boundary(geometry_utilities,
                                                                      10 * num_rat_points * solver.problem.num_vertices,
                                                                      solver,
                                                                      boundary_conditions,
                                                                      boundary_tangent_derivatives=boundary_tangent_derivatives)

        self.assertLess(err_max_l2, b=1.0e-8)
        self.assertLess(err_max_h1, b=1.0e-3)

        if plot_function:
            print('Max error L2 {:<.16e} - Max Error H1 {:<.16e} on test points on the boundary.'
                  .format(err_max_l2, err_max_h1))

            plot_points_boundary_distributions(solver.domain_vertices(), solver.flat_poles, solver.boundary_points)

            plot_function_and_tangent_derivatives_on_edges(geometry_utilities,
                                                           10 * num_rat_points *
                                                           solver.problem.num_vertices,
                                                           solver,
                                                           boundary_conditions,
                                                           boundary_tangent_derivatives=boundary_tangent_derivatives)


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

        def boundary_tangent_derivatives(v: int, vertices: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:

            if v == 4:
                return (np.ones(points.shape[1])) / np.linalg.norm(vertices[:, 0:1] - vertices[:, 4:5])
            elif v == 0:
                return (-np.ones(points.shape[1])) / np.linalg.norm(vertices[:, 0:1] - vertices[:, 1:2])
            else:
                return 0.0 + 0.0 * points[1, :]

        problem = LaplaceProblem(domain_vertices, domain_scale, internal_point, boundary_conditions)

        solver = LaplaceSolver(geometry_utilities, problem, iterations, list_poles_vertices)

        num_rat_points = 30
        err_max_l2, err_max_h1 = evaluate_accuracy_on_domain_boundary(geometry_utilities,
                                                                      10 * num_rat_points * solver.problem.num_vertices,
                                                                      solver,
                                                                      boundary_conditions,
                                                                      boundary_tangent_derivatives=boundary_tangent_derivatives)

        self.assertLess(err_max_l2, b=1.0e-8)
        self.assertLess(err_max_h1, b=1.0e-5)

        if plot_function:
            print('Max error L2 {:<.16e} - Max Error H1 {:<.16e} on test points on the boundary.'
                  .format(err_max_l2, err_max_h1))

            plot_points_boundary_distributions(solver.domain_vertices(), solver.flat_poles, solver.boundary_points)

            plot_function_and_tangent_derivatives_on_edges(geometry_utilities,
                                                           10 * num_rat_points *
                                                           solver.problem.num_vertices,
                                                           solver,
                                                           boundary_conditions,
                                                           boundary_tangent_derivatives=boundary_tangent_derivatives)


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

        def boundary_tangent_derivatives(v: int, vertices: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:

            if v == 7:
                return (np.ones(points.shape[1])) / np.linalg.norm(vertices[:, 7:8] - vertices[:, 0:1])
            elif v == 0:
                return (-np.ones(points.shape[1])) / np.linalg.norm(vertices[:, 0:1] - vertices[:, 1:2])
            else:
                return 0.0 + 0.0 * points[1, :]

        problem = LaplaceProblem(domain_vertices, domain_scale, internal_point, boundary_conditions)

        solver = LaplaceSolver(geometry_utilities, problem, iterations, list_poles_vertices)

        num_rat_points = 30
        err_max_l2, err_max_h1 = evaluate_accuracy_on_domain_boundary(geometry_utilities,
                                                                      10 * num_rat_points * solver.problem.num_vertices,
                                                                      solver,
                                                                      boundary_conditions,
                                                                      boundary_tangent_derivatives=boundary_tangent_derivatives)

        self.assertLess(err_max_l2, b=1.0e-2)
        self.assertLess(err_max_h1, b=1.0e-2)


        if plot_function:
            print('Max error L2 {:<.16e} - Max Error H1 {:<.16e} on test points on the boundary.'
                  .format(err_max_l2, err_max_h1))

            plot_points_boundary_distributions(solver.domain_vertices(), solver.flat_poles, solver.boundary_points)

            plot_function_and_tangent_derivatives_on_edges(geometry_utilities,
                                                           10 * num_rat_points *
                                                           solver.problem.num_vertices,
                                                           solver,
                                                           boundary_conditions,
                                                           boundary_tangent_derivatives=boundary_tangent_derivatives)

if __name__ == '__main__':

    unittest.main()

