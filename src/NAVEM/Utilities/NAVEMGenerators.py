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

from NAVEM.Utilities.HarmonicPolynomials import HarmonicPolynomials, change_basis_matrix
from NAVEM.Utilities.LaplaceSolver import hanging_function
from NAVEM.Utilities.FunctionUtilities import *
from NAVEM.PCC_2D.NAVEM_Data_PCC_2D import NAVEMElementType
from NAVEM.Utilities.RationalFunction import RationalFunction


class NAVEMGenerators:

    def __init__(self, geometry_utilities: gedim.GeometryUtilities,
                 num_vertices: int,
                 harmonic_degree: int,
                 use_hanging_function: bool,
                 normalization_diameter: float,
                 num_rationals_points: int,
                 rational_type_function: RationalFunction.RationalType,
                 element_type: NAVEMElementType):

        self.normalization_diameter = normalization_diameter
        self.normalization_centroid = np.zeros([3, 1])
        self.geometry_utilities = geometry_utilities
        self.num_vertices = num_vertices

        # Harmonic polynomials
        self.harmonic_degree = harmonic_degree
        self.harmonic_polynomials = HarmonicPolynomials(self.harmonic_degree, HarmonicPolynomials.HarmonicType.total)

        self.num_harmonic_polynomials = self.harmonic_polynomials.num_harmonic_polynomials

        # define points on the square centered in the origin with length edge = normalization_diameter
        num_1d_pts = int(np.ceil(np.sqrt(self.num_harmonic_polynomials)) + 1)
        change_basis_1d_pts = np.linspace(- 0.5 * self.normalization_diameter,
                                          0.5 * self.normalization_diameter, num_1d_pts)
        change_basis_xs = np.tile(change_basis_1d_pts, num_1d_pts)
        change_basis_ys = np.repeat(change_basis_1d_pts, num_1d_pts)

        # x on the first row, while y on the second row
        change_basis_points = np.stack([change_basis_xs, change_basis_ys], axis=0)

        harmonic_vandermonde = self.harmonic_polynomials.vander(change_basis_points, self.normalization_diameter,
                                                                self.normalization_centroid)[0]

        self.change_basis_matrix = change_basis_matrix(harmonic_vandermonde)
        self.num_generators = self.num_harmonic_polynomials


        # Hanging functions
        self.use_hanging_function = use_hanging_function
        self.list_id_vertices_hanging: List[int] = []

        if self.use_hanging_function:

            match element_type:
                case NAVEMElementType.generic_convex:
                    self.list_id_vertices_hanging = [0, 1, num_vertices - 1]
                case NAVEMElementType.generic_concave:
                    self.list_id_vertices_hanging = [i for i in range(num_vertices)]
                case _:
                    raise ValueError("Not valid element type")

            self.hanging_function = hanging_function(geometry_utilities)
            self.num_hanging_functions = len(self.list_id_vertices_hanging)
            self.num_generators += self.num_hanging_functions

        # Rational functions
        self.num_rationals_points = num_rationals_points
        self.rational_type_function = rational_type_function
        self.list_id_vertices_rationals: List[int] = []

        if self.num_rationals_points > 0:

            match element_type:
                case NAVEMElementType.generic_convex:
                    self.list_id_vertices_rationals = [0, 1, num_vertices - 1]
                case NAVEMElementType.generic_concave:
                    self.list_id_vertices_rationals = [i for i in range(num_vertices)]
                case _:
                    raise ValueError("Not valid element type")

            self.rational_function = RationalFunction(self.rational_type_function)
            self.num_rational_functions = (self.rational_function.
                                           num_rational_functions(self.num_rationals_points *
                                                                  len(self.list_id_vertices_rationals),
                                                                  self.rational_type_function))
            self.num_generators += self.num_rational_functions

    def vander_and_vander_derivatives(self, points: NDArray[np.float64], polygon_vertices: NDArray[np.float64],
                                      internal_angles: NDArray[np.float64] = np.zeros(0),
                                      vertex_distance: List[float] = None) -> Tuple[
        NDArray[np.float64], NDArray[np.float64]]:

        harm_vandermonde, total_harm_vandermonde = self.harmonic_polynomials.vander(points, self.normalization_diameter,
                                                                                    self.normalization_centroid)

        grad_harm_vandermonde = self.harmonic_polynomials.vander_derivatives(total_harm_vandermonde,
                                                                             self.normalization_diameter)

        grad_vandermonde = np.zeros([2, points.shape[1], self.num_generators])

        grad_vandermonde[0, :, 0:self.num_harmonic_polynomials] = grad_harm_vandermonde[0, :,
                                                                  :] @ self.change_basis_matrix
        grad_vandermonde[1, :, 0:self.num_harmonic_polynomials] = grad_harm_vandermonde[1, :,
                                                                  :] @ self.change_basis_matrix
        vandermonde = harm_vandermonde @ self.change_basis_matrix

        if self.use_hanging_function:

            for v in range(len(self.list_id_vertices_hanging)):
                v_id = self.list_id_vertices_hanging[v]

                translation_1, translation_2, b_matrix, b_matrix_inv = compute_map_f(self.geometry_utilities,
                                                                                     v_id, polygon_vertices,
                                                                                     self.hanging_function.domain_vertices(),
                                                                                     np.squeeze(
                                                                                         internal_angles).tolist(),
                                                                                     vertex_distance)

                original_points, _ = map_f_inv(translation_1, translation_2, b_matrix, b_matrix_inv, points)
                jac = b_matrix_inv.T
                hang_vandermonde, grad_hang_vandermonde = self.hanging_function.vander_and_vander_derivatives(
                    original_points)

                grad_hang_vandermonde = map_vander_derivatives(grad_hang_vandermonde, jac)

                grad_vandermonde[0, :, (self.num_harmonic_polynomials + v):self.num_harmonic_polynomials + v + 1] \
                    = grad_hang_vandermonde[0, :, :]
                grad_vandermonde[1, :, (self.num_harmonic_polynomials + v):self.num_harmonic_polynomials + v + 1] \
                    = grad_hang_vandermonde[1, :, :]

                vandermonde = np.concatenate((vandermonde, hang_vandermonde), axis=1)

        if self.num_rationals_points > 0:

            polygon_edge_normals = self.geometry_utilities.polygon_edge_normals(polygon_vertices)
            polygon_vertex_bisectors = compute_polygon_external_bisectors(self.geometry_utilities, polygon_edge_normals)
            polygon_bounding_box = self.geometry_utilities.points_bounding_box(polygon_vertices)
            polygon_scale = max([polygon_bounding_box[0, 1] - polygon_bounding_box[0, 0],
                                 polygon_bounding_box[1, 1] - polygon_bounding_box[1, 0]])

            poles, flat_poles = RationalFunction.compute_poles(self.geometry_utilities, self.num_rationals_points,
                                                               self.list_id_vertices_rationals, polygon_vertices,
                                                               np.squeeze(internal_angles).tolist(),
                                                               polygon_vertex_bisectors, polygon_scale)

            dist = RationalFunction.compute_distance_pole_vertex(self.list_id_vertices_rationals, poles, flat_poles, polygon_vertices)



            n_hangings = self.num_hanging_functions
            grad_rat_vandermonde = self.rational_function.vander_derivatives(points, flat_poles, dist)

            grad_vandermonde[0, :, (self.num_harmonic_polynomials + n_hangings):
                                   (self.num_harmonic_polynomials + n_hangings + self.num_rational_functions)] \
                = grad_rat_vandermonde[0, :, :]
            grad_vandermonde[1, :, (self.num_harmonic_polynomials + n_hangings):
                                   (self.num_harmonic_polynomials + n_hangings + self.num_rational_functions)] \
                = grad_rat_vandermonde[1, :, :]

            vandermonde_rat = self.rational_function.vander(points, flat_poles, dist)
            vandermonde = np.concatenate((vandermonde, vandermonde_rat), axis=1)

        return vandermonde, grad_vandermonde
