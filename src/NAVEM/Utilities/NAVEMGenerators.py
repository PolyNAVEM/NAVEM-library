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


class NAVEMGenerators:

    def __init__(self, geometry_utilities: gedim.GeometryUtilities,
                 num_vertices: int,
                 harmonic_degree: int,
                 use_hanging_function: bool,
                 normalization_diameter: float):

        self.normalization_diameter = normalization_diameter
        self.normalization_centroid = np.zeros([3, 1])
        self.geometry_utilities = geometry_utilities
        self.num_vertices = num_vertices

        self.harmonic_degree = harmonic_degree
        self.harmonic_polynomials = HarmonicPolynomials(self.harmonic_degree, HarmonicPolynomials.HarmonicType.real)

        self.use_hanging_function = use_hanging_function
        self.list_id_vertices_hanging: List[int] = []
        if use_hanging_function:
            self.list_id_vertices_hanging = [0, 1, num_vertices - 1]


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

        if self.use_hanging_function:
            self.hanging_function = hanging_function(geometry_utilities)
            self.num_hanging_functions = len(self.list_id_vertices_hanging)
            self.num_generators += self.num_hanging_functions


    def vander_and_vander_derivatives(self, points: NDArray[np.float64], polygon_vertices: NDArray[np.float64],
                                      internal_angles: NDArray[np.float64] =np.zeros(0),
                                      vertex_distance: List[float] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        harm_vandermonde, total_harm_vandermonde = self.harmonic_polynomials.vander(points, self.normalization_diameter,
                                                                                    self.normalization_centroid)

        grad_harm_vandermonde = self.harmonic_polynomials.vander_derivatives(total_harm_vandermonde,
                                                                             self.normalization_diameter)

        grad_vandermonde = np.zeros([2, points.shape[1], self.num_generators])


        grad_vandermonde[0, :, 0:self.num_harmonic_polynomials] = grad_harm_vandermonde[0, :, :] @ self.change_basis_matrix
        grad_vandermonde[1, :, 0:self.num_harmonic_polynomials] = grad_harm_vandermonde[1, :, :] @ self.change_basis_matrix
        vandermonde = harm_vandermonde @ self.change_basis_matrix


        if self.use_hanging_function:

            for v in range(len(self.list_id_vertices_hanging)):
                v_id = self.list_id_vertices_hanging[v]

                translation_1, translation_2, b_matrix, b_matrix_inv = compute_map_f(self.geometry_utilities,
                                                                                     v_id, polygon_vertices,
                                                                                     self.hanging_function.domain_vertices(),
                                                                                     np.squeeze(internal_angles).tolist(),
                                                                                     vertex_distance)

                original_points, _ = map_f_inv(translation_1, translation_2, b_matrix, b_matrix_inv, points)
                jac = b_matrix_inv.T
                hang_vandermonde, grad_hang_vandermonde = self.hanging_function.vander_and_vander_derivatives(original_points)

                grad_hang_vandermonde = map_vander_derivatives(grad_hang_vandermonde, jac)

                grad_vandermonde[0, :, (self.num_harmonic_polynomials+v):self.num_harmonic_polynomials+v+1] \
                    = grad_hang_vandermonde[0, :, :]
                grad_vandermonde[1, :, (self.num_harmonic_polynomials+v):self.num_harmonic_polynomials+v+1] \
                    = grad_hang_vandermonde[1, :, :]

                vandermonde = np.concatenate((vandermonde, hang_vandermonde), axis=1)

        return vandermonde, grad_vandermonde
