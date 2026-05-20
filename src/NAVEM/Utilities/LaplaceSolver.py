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

import importlib

from scipy.sparse import diags
from NAVEM.Utilities.HarmonicPolynomials import HarmonicPolynomials
from NAVEM.Utilities.RationalFunction import RationalFunction
from NAVEM.geometry.geometry_utilities import *
from NAVEM.Utilities.FunctionUtilities import Function
import matplotlib.pyplot as plt
from matplotlib import use
from typing import Tuple, Callable
from NAVEM.Utilities.points_generator import reference_points_distribution, PointsSegmentDistributionType

try:
    use("Qt5Agg")
    importlib.reload(plt)
except ImportError:
    neglectingGraphicPackages = True


class LaplaceProblem:

    def __init__(self, domain_vertices: NDArray[np.float64], domain_scale: float, internal_point: NDArray[np.float64],
                 boundary_dirichlet_conditions: Callable[[int, NDArray[np.float64]], NDArray[np.float64]]):
        self.domain_vertices = domain_vertices
        self.num_vertices = self.domain_vertices.shape[1]
        self.domain_scale = domain_scale
        self.internal_point = internal_point
        self.boundary_dirichlet_conditions = boundary_dirichlet_conditions


class LaplaceSolver(Function):

    def __init__(self, geometry_utilities: gedim.GeometryUtilities, problem: LaplaceProblem,
                 iterations: List[int], list_poles_vertices: List[int], error_tolerance=1.0e-6,
                 harmonic_type: HarmonicPolynomials.HarmonicType = HarmonicPolynomials.HarmonicType.total,
                 rational_type: RationalFunction.RationalType = RationalFunction.RationalType.total,
                 use_weights: bool = True):

        self.geometry_utilities = geometry_utilities
        self.problem = problem
        self.error_tolerance = error_tolerance
        self.error = 1.0
        self.use_weights = use_weights
        self.harmonic_type = harmonic_type
        self.rational_type = rational_type
        self.iterations = iterations
        self.num_actual_iteration = iterations[0]
        self.list_poles_vertices = list_poles_vertices

        self.harm_deg = 0
        self.harmonic_polynomials = HarmonicPolynomials(self.harm_deg, self.harmonic_type)

        self.rational_functions = RationalFunction(self.rational_type)

        self.polygon_interior_angles = compute_polygon_interior_angles(self.problem.domain_vertices)
        self.polygon_edge_normals = self.geometry_utilities.polygon_edge_normals(self.problem.domain_vertices)
        self.polygon_vertex_bisectors = compute_polygon_external_bisectors(geometry_utilities,
                                                                           self.polygon_edge_normals)

        self.flat_poles = np.zeros([3, 0])
        self.coefficients = np.zeros([0])
        self.distance_from_poles = np.zeros([0])
        self.boundary_points = np.zeros([3, 0])

        self.solve()

    def domain_vertices(self) -> NDArray[np.float64]:
        return self.problem.domain_vertices

    def vander(self, points: NDArray[np.float64]) -> NDArray[np.float64]:

        vandermonde_harmonic, _ = self.harmonic_polynomials.vander(points,
                                                                   self.problem.domain_scale,
                                                                   self.problem.internal_point)

        vandermonde_rational = self.rational_functions.vander(points, self.flat_poles, self.distance_from_poles)

        matrix_coefficients = np.concatenate([vandermonde_harmonic, vandermonde_rational], axis=1)
        vandermonde = np.expand_dims(matrix_coefficients @ self.coefficients, axis=1)

        return vandermonde

    def vander_derivatives(self, points: NDArray[np.float64]) -> NDArray[np.float64]:

        _, vandermonde_total_harm = self.harmonic_polynomials.vander(points,
                                                                     self.problem.domain_scale,
                                                                     self.problem.internal_point)

        grad_vandermonde_harm = self.harmonic_polynomials.vander_derivatives(vandermonde_total_harm,
                                                                             self.problem.domain_scale)
        grad_vandermonde_rat = self.rational_functions.vander_derivatives(points, self.flat_poles,
                                                                          self.distance_from_poles)

        grad_vandermonde = np.zeros([2, points.shape[1], 1])
        grad_vandermonde[0, :, :] = np.expand_dims(np.concatenate([grad_vandermonde_harm[0, :, :],
                                                                   grad_vandermonde_rat[0, :, :]], axis=1)
                                                   @ self.coefficients, axis=1)
        grad_vandermonde[1, :, :] = np.expand_dims(np.concatenate([grad_vandermonde_harm[1, :, :],
                                                                   grad_vandermonde_rat[1, :, :]], axis=1)
                                                   @ self.coefficients, axis=1)

        return grad_vandermonde

    def vander_and_vander_derivatives(self, points: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64], NDArray[np.float64]]:

        vandermonde_harmonic, vandermonde_total_harm \
            = self.harmonic_polynomials.vander(points,
                                               self.problem.domain_scale,
                                               self.problem.internal_point)

        grad_vandermonde_harm = self.harmonic_polynomials.vander_derivatives(vandermonde_total_harm,
                                                                             self.problem.domain_scale)

        grad_vandermonde_rat = self.rational_functions.vander_derivatives(points, self.flat_poles,
                                                                          self.distance_from_poles)
        vandermonde_rational = self.rational_functions.vander(points, self.flat_poles, self.distance_from_poles)

        matrix_coefficients = np.concatenate([vandermonde_harmonic, vandermonde_rational], axis=1)
        vandermonde = np.expand_dims(matrix_coefficients @ self.coefficients, axis=1)

        grad_vandermonde = np.zeros([2, points.shape[1], 1])
        grad_vandermonde[0, :, :] = np.expand_dims(np.concatenate([grad_vandermonde_harm[0, :, :],
                                                                   grad_vandermonde_rat[0, :, :]], axis=1)
                                                   @ self.coefficients, axis=1)
        grad_vandermonde[1, :, :] = np.expand_dims(np.concatenate([grad_vandermonde_harm[1, :, :],
                                                                   grad_vandermonde_rat[1, :, :]], axis=1)
                                                   @ self.coefficients, axis=1)

        return vandermonde, grad_vandermonde

    def compute_boundary_sample_points_and_right_term(self,
                                                      poles: List[NDArray[np.float64]],
                                                      num_total_poles: int) -> Tuple[
        NDArray[np.float64], NDArray[np.float64]]:

        """

        Compute boundary evaluation points if, given a vertex v, we have a set of associated poles, for each pole k
        we consider 3 boundary points located at distance (1/3)*distance_{v,k}, (2/3)*distance_{v,k}, and distance_{v,k}
        Otherwise we consider uniform points.

        :param poles:
        :param num_total_poles:
        :return: boundary_points and the right hand side
        """

        boundary_points = []
        right_term = []

        for v in range(self.problem.num_vertices):

            prev_id = v - 1 if v > 0 else self.problem.num_vertices - 1
            previous_tangent_edge = self.problem.domain_vertices[:, prev_id] - self.problem.domain_vertices[:, v]
            tangent_edge = (self.problem.domain_vertices[:, (v + 1) % self.problem.num_vertices]
                            - self.problem.domain_vertices[:, v])
            length_previous_edge = np.linalg.norm(previous_tangent_edge)
            length_edge = np.linalg.norm(tangent_edge)

            if v in self.list_poles_vertices:

                p = self.list_poles_vertices.index(v)
                num_poles = poles[p].shape[1]

                for j in range(num_poles):

                    dist = np.linalg.norm(self.problem.domain_vertices[:, v] - poles[p][:, j])

                    for k in range(3):
                        t = (k + 1) / 3 * dist
                        if t < length_previous_edge:
                            bbd = (t * previous_tangent_edge / length_previous_edge
                                   + self.problem.domain_vertices[:, v])

                            boundary_points.append(bbd)
                            right_term.append(
                                self.problem.boundary_dirichlet_conditions(prev_id, np.expand_dims(bbd, axis=1))[0])

                        if t < length_edge:
                            bbd = (t * tangent_edge / length_edge
                                   + self.problem.domain_vertices[:, v])

                            boundary_points.append(bbd)
                            right_term.append(
                                self.problem.boundary_dirichlet_conditions(v, np.expand_dims(bbd, axis=1))[0])

            else:
                num_points = int(num_total_poles / len(self.list_poles_vertices))
                reference_points = reference_points_distribution(0.0, 0.5, num_points,
                                                                 PointsSegmentDistributionType.exponential)[1:-1]
                for j in range(len(reference_points)):
                    bbd = self.problem.domain_vertices[:, v] + reference_points[j] * previous_tangent_edge
                    boundary_points.append(bbd)
                    right_term.append(
                        self.problem.boundary_dirichlet_conditions(prev_id, np.expand_dims(bbd, axis=1))[0])

                    bbd = self.problem.domain_vertices[:, v] + reference_points[j] * tangent_edge
                    boundary_points.append(bbd)
                    right_term.append(self.problem.boundary_dirichlet_conditions(v, np.expand_dims(bbd, axis=1))[0])

        boundary_points = np.array(boundary_points).T
        right_term = np.array(right_term).T

        return boundary_points, right_term

    def solve(self):

        for n in self.iterations:

            self.num_actual_iteration = n

            # Exponential clustered poles at corners
            poles, self.flat_poles = self.rational_functions.compute_poles(self.geometry_utilities, int(n),
                                                                           self.list_poles_vertices,
                                                                           self.problem.domain_vertices,
                                                                           self.polygon_interior_angles,
                                                                           self.polygon_vertex_bisectors,
                                                                           self.problem.domain_scale)

            self.distance_from_poles = (
                self.rational_functions.compute_distance_pole_vertex(self.list_poles_vertices,
                                                                     poles,
                                                                     self.flat_poles,
                                                                     self.problem.domain_vertices))

            # Distance of poles from corners and boundary sample points
            self.boundary_points, right_term = (
                self.compute_boundary_sample_points_and_right_term(poles, self.flat_poles.shape[1]))

            num_boundary_points = self.boundary_points.shape[1]
            num_functions = right_term.shape[0]

            # Polynomial approximation
            self.harm_deg = int(np.ceil(n / 2))

            self.harmonic_polynomials = HarmonicPolynomials(self.harm_deg, self.harmonic_type)
            vandermonde_harmonic, _ = self.harmonic_polynomials.vander(self.boundary_points,
                                                                       self.problem.domain_scale,
                                                                       self.problem.internal_point)

            vandermonde_rational = self.rational_functions.vander(self.boundary_points, self.flat_poles,
                                                                  self.distance_from_poles)

            matrix_coefficient = np.concatenate([vandermonde_harmonic, vandermonde_rational], axis=1)

            weights = np.ones([1, num_boundary_points])
            if self.use_weights:
                weights = np.linalg.norm(self.boundary_points - self.problem.domain_vertices[:, 0:1], axis=0)
                for v in range(self.problem.num_vertices - 1):
                    weights = np.minimum(weights, np.linalg.norm(self.boundary_points
                                                                 - self.problem.domain_vertices[:, (v + 1):(v + 2)],
                                                                 axis=0))

            weights = diags([weights], [0], shape=(num_functions, num_functions)).toarray()

            self.coefficients = np.linalg.lstsq(matrix_coefficient, right_term, rcond=None)[0]
            error_vec = weights @ (matrix_coefficient @ self.coefficients - right_term)
            self.error = np.linalg.norm(error_vec, ord=np.inf)

            if self.error < self.error_tolerance:
                break


def hanging_function(geometry_utilities: gedim.GeometryUtilities) -> LaplaceSolver:
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

    return solver
