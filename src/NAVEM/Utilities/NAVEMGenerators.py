import numpy as np
from pypolydim import gedim
from typing import List
from src.NAVEM.Utilities.HarmonicPolynomials import HarmonicPolynomials
from src.NAVEM.Utilities.LaplaceSolver import hanging_function

class NAVEMGenerators:

    def __init__(self, geometry_utilities: gedim.GeometryUtilities,
                 harmonic_degree: int,
                 use_hanging_function: bool,
                 list_id_vertices_hanging: List[int],
                 normalization_diameter):

        self.normalization_diameter = normalization_diameter
        self.normalization_centroid = np.zeros([3, 1])
        self.geometry_utilities = geometry_utilities
        
        self.harmonic_degree = harmonic_degree
        self.harmonic_polynomials = HarmonicPolynomials(self.harmonic_degree, HarmonicPolynomials.HarmonicType.real)

        self.use_hanging_function = use_hanging_function
        self.list_id_vertices_hanging = list_id_vertices_hanging

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



        self.change_basis_matrix = pol_utils.change_basis_matrix(harmonic_vandermonde)

        self.num_generators = self.num_harmonic_polynomials

        if self.use_hanging_function:
            self.hanging_function = hanging_function(geometry_utilities)
            self.num_hanging_functions = len(self.list_id_vertices_hanging)
            self.num_generators += self.num_hanging_functions


    def vander(self, points, polygon_vertices, flat_poles=np.zeros(0), dist=np.zeros(0),
                                      internal_angles=np.zeros(0), vertex_distance=[]):

        harmonic_vandermonde = (
                self.harmonic_polynomials.vander(points,
                                                 self.normalization_diameter,
                                                 self.normalization_centroid)[0] @ self.change_basis_matrix)

        if self.use_hanging_function:
            for v in range(len(self.list_id_vertices_hanging)):
                v_id = self.list_id_vertices_hanging[v]
                translation_1, translation_2, B, B_inv = HangingFunctionUtilities.compute_map_f(self.geometry_utilities,
                                                                                                v_id, polygon_vertices,
                                                                                                self.hanging_function.polygon_vertices,
                                                                                                internal_angles,
                                                                                                vertex_distance)

                original_points, _ = HangingFunctionUtilities.map_f_inv(translation_1, translation_2, B, B_inv, points)

                hang_vandermonde = self.hanging_function.vander(original_points)
                vandermonde = np.concatenate((vandermonde, hang_vandermonde), axis=1)

        return vandermonde

    def vander_derivatives(self, points, polygon_vertices, flat_poles=np.zeros(0), dist=np.zeros(0),
                           internal_angles=np.zeros(0), vertex_distance=[]):

        _, harm_vandermonde = self.harmonic_polynomials.vander(points, self.normalization_diameter,
                                                               self.normalization_centroid)

        grad_harm_vandermonde = self.harmonic_polynomials.vander_derivatives(harm_vandermonde,
                                                                             self.normalization_diameter)

        grad_vandermonde = np.zeros([2, points.shape[1], self.n_tot_polynomial])

        if self.vem_order > 1:
            mon_vandermonde = self.monomials.vandermonde(points, self.normalization_diameter, self.normalization_centroid)

            grad_mon_vandermonde = self.monomials.compute_derivatives_monomials(mon_vandermonde,
                                                                                self.normalization_diameter)

            grad_pol_vandermonde = np.zeros([2, points.shape[1], self.n_polynomial])
            grad_pol_vandermonde[0, :, :] = np.concatenate((grad_mon_vandermonde[0, :, :],
                                                            grad_harm_vandermonde[0, :, 1+2*self.vem_order:]),
                                                           axis=1) @ self.change_basis_matrix
            grad_pol_vandermonde[1, :, :] = np.concatenate((grad_mon_vandermonde[1, :, :],
                                                            grad_harm_vandermonde[1, :, 1+2*self.vem_order:]),
                                                           axis=1) @ self.change_basis_matrix

            grad_vandermonde[0, :, 0:self.n_polynomial] = grad_pol_vandermonde[0, :, :]
            grad_vandermonde[1, :, 0:self.n_polynomial] = grad_pol_vandermonde[1, :, :]
        else:
            grad_vandermonde[0, :, 0:self.n_polynomial] = grad_harm_vandermonde[0, :, :] @ self.change_basis_matrix
            grad_vandermonde[1, :, 0:self.n_polynomial] = grad_harm_vandermonde[1, :, :] @ self.change_basis_matrix



        if self.use_hanging_function:
            for v in range(len(self.list_id_vertices_hanging)):
                v_id = self.list_id_vertices_hanging[v]

                translation_1, translation_2, B, B_inv = HangingFunctionUtilities.compute_map_f(self.geometry_utilities,
                                                                                                v_id, polygon_vertices,
                                                                                                self.hanging_function.polygon_vertices,
                                                                                                internal_angles,
                                                                                                vertex_distance)

                original_points, _ = HangingFunctionUtilities.map_f_inv(translation_1, translation_2, B, B_inv, points)
                jac = B_inv.T
                grad_hang_vandermonde = HangingFunctionUtilities.compute_vander_derivatives(original_points,
                                                                                            jac,
                                                                                            self.hanging_function)

                grad_vandermonde[0, :, (self.n_polynomial+v):self.n_polynomial+v+1] \
                    = grad_hang_vandermonde[0, :, :]
                grad_vandermonde[1, :, (self.n_polynomial+v):self.n_polynomial+v+1] \
                    = grad_hang_vandermonde[1, :, :]


        if self.use_rational_function:
            n_hangings = self.num_hanging_functions
            grad_rat_vandermonde = self.rational_function.vander_derivatives(points, flat_poles, dist)

            grad_vandermonde[0, :, (self.n_polynomial + n_hangings ):
                                   (self.n_polynomial + n_hangings + self.num_rational_functions)] \
                = grad_rat_vandermonde[0, :, :]
            grad_vandermonde[1, :, (self.n_polynomial + n_hangings ):
                                   (self.n_polynomial + n_hangings + self.num_rational_functions)] \
                = grad_rat_vandermonde[1, :, :]

        return grad_vandermonde

    def vander_and_vander_derivatives(self, points, polygon_vertices, flat_poles=np.zeros(0), dist=np.zeros(0),
                                      internal_angles=np.zeros(0), vertex_distance=[]):

        harm_vandermonde, total_harm_vandermonde = self.harmonic_polynomials.vander(points, self.normalization_diameter,
                                                                                    self.normalization_centroid)

        grad_harm_vandermonde = self.harmonic_polynomials.vander_derivatives(total_harm_vandermonde,
                                                                             self.normalization_diameter)

        grad_vandermonde = np.zeros([2, points.shape[1], self.n_tot_polynomial])

        if self.vem_order > 1:
            mon_vandermonde = self.monomials.vandermonde(points, self.normalization_diameter, self.normalization_centroid)

            grad_mon_vandermonde = self.monomials.compute_derivatives_monomials(mon_vandermonde,
                                                                                self.normalization_diameter)

            grad_pol_vandermonde = np.zeros([2, points.shape[1], self.n_polynomial])
            grad_pol_vandermonde[0, :, :] = np.concatenate((grad_mon_vandermonde[0, :, :],
                                                            grad_harm_vandermonde[0, :, 1+2*self.vem_order:]),
                                                           axis=1) @ self.change_basis_matrix
            grad_pol_vandermonde[1, :, :] = np.concatenate((grad_mon_vandermonde[1, :, :],
                                                            grad_harm_vandermonde[1, :, 1+2*self.vem_order:]),
                                                           axis=1) @ self.change_basis_matrix
            grad_vandermonde[0, :, 0:self.n_polynomial] = grad_pol_vandermonde[0, :, :]
            grad_vandermonde[1, :, 0:self.n_polynomial] = grad_pol_vandermonde[1, :, :]

            vandermonde = np.concatenate((mon_vandermonde, harm_vandermonde[:, 1 + 2 * self.vem_order:]),
                                         axis=1) @ self.change_basis_matrix
        else:
            grad_vandermonde[0, :, 0:self.n_polynomial] = grad_harm_vandermonde[0, :, :] @ self.change_basis_matrix
            grad_vandermonde[1, :, 0:self.n_polynomial] = grad_harm_vandermonde[1, :, :] @ self.change_basis_matrix

            vandermonde = harm_vandermonde @ self.change_basis_matrix


        if self.use_hanging_function:

            for v in range(len(self.list_id_vertices_hanging)):
                v_id = self.list_id_vertices_hanging[v]

                translation_1, translation_2, B, B_inv = HangingFunctionUtilities.compute_map_f(self.geometry_utilities,
                                                                                                v_id, polygon_vertices,
                                                                                                self.hanging_function.polygon_vertices,
                                                                                                internal_angles,
                                                                                                vertex_distance)

                original_points, _ = HangingFunctionUtilities.map_f_inv(translation_1, translation_2, B, B_inv, points)
                jac = B_inv.T
                hang_vandermonde, grad_hang_vandermonde = self.hanging_function.vander_and_vander_derivatives(original_points)

                grad_hang_vandermonde = HangingFunctionUtilities.map_vander_derivatives(grad_hang_vandermonde, jac)

                grad_vandermonde[0, :, (self.n_polynomial+v):self.n_polynomial+v+1] \
                    = grad_hang_vandermonde[0, :, :]
                grad_vandermonde[1, :, (self.n_polynomial+v):self.n_polynomial+v+1] \
                    = grad_hang_vandermonde[1, :, :]

                vandermonde = np.concatenate((vandermonde, hang_vandermonde), axis=1)

        if self.use_rational_function:
            n_hangings = self.num_hanging_functions
            grad_rat_vandermonde = self.rational_function.vander_derivatives(points, flat_poles, dist)

            grad_vandermonde[0, :, (self.n_polynomial + n_hangings):
                                   (self.n_polynomial + n_hangings + self.num_rational_functions)] \
                = grad_rat_vandermonde[0, :, :]
            grad_vandermonde[1, :, (self.n_polynomial + n_hangings):
                                   (self.n_polynomial + n_hangings + self.num_rational_functions)] \
                = grad_rat_vandermonde[1, :, :]

            vandermonde_rat = self.rational_function.vander(points, flat_poles, dist)
            vandermonde = np.concatenate((vandermonde, vandermonde_rat), axis=1)

        return vandermonde, grad_vandermonde


class EmptyHangingFunction:
    def __init__(self):
        self.harm_deg = 0
        self.flat_poles = np.array([]).reshape([0, 0])
        self.function_type = 0

