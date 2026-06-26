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

import numpy as np
from enum import Enum
from pypolydim import gedim
from typing import List, Tuple
from numpy.typing import NDArray


class RationalFunction:
    class RationalType(Enum):
        total = 1
        real = 2
        imag = 3

    def __init__(self, vander_type: RationalType = RationalType.real):

        self.vander_type = vander_type

    @staticmethod
    def compute_poles(geometry_utilities: gedim.GeometryUtilities, num_poles: int, list_pole_vertices: List[int],
                      polygon_vertices: NDArray[np.float64], polygon_interior_angles: List[float],
                      polygon_vertex_bisectors: NDArray[np.float64], polygon_scale: float, sigma: float = 4.0) \
            -> Tuple[List[NDArray[np.float64]], NDArray[np.float64]]:

        """

        The kth Pole is defined throughout the distance_fun_k = exp(-sigma * (sqrt{num_poles} - sqrt{k}))
        and they are distributed along the external bisector to the related vertex

        :param geometry_utilities:
        :param num_poles:
        :param list_pole_vertices:
        :param polygon_vertices:
        :param polygon_interior_angles:
        :param polygon_vertex_bisectors:
        :param polygon_scale:
        :param sigma:
        :return:
        """

        poles = []
        num_total_poles = 0

        for v in list_pole_vertices:

            n = num_poles
            if polygon_interior_angles[v] > np.pi:
                n = 3 * num_poles

            distance_fun = np.exp(-sigma * (np.sqrt(n) - np.sqrt(np.arange(1, n + 1))))
            pole = []
            for i in range(len(distance_fun)):
                pp = (polygon_vertices[:, v] +
                      polygon_scale * distance_fun[i] * polygon_vertex_bisectors[:, v])

                point_polygon_position: gedim.GeometryUtilities.PointPolygonPositionResult \
                    = geometry_utilities.point_polygon_position_ray_casting(pp, polygon_vertices)
                match point_polygon_position.type:
                    case gedim.GeometryUtilities.PointPolygonPositionResult.Types.unknown:
                        pass
                    case gedim.GeometryUtilities.PointPolygonPositionResult.Types.outside:
                        pole.append(pp)
                    case gedim.GeometryUtilities.PointPolygonPositionResult.Types.border_edge:
                        pass
                    case gedim.GeometryUtilities.PointPolygonPositionResult.Types.border_vertex:
                        pass
                    case gedim.GeometryUtilities.PointPolygonPositionResult.Types.inside:
                        pass
                    case _:
                        raise ValueError("not valid point polygon position type")

            pole = np.array(pole).T
            poles.append(pole)

            num_total_poles += pole.shape[1]

        flat_poles = np.zeros([3, num_total_poles])
        offset = 0
        for p in range(len(list_pole_vertices)):
            flat_poles[:, offset:offset + poles[p].shape[1]] = poles[p]
            offset += poles[p].shape[1]

        return poles, flat_poles

    @staticmethod
    def compute_distance_pole_vertex(list_pole_vertices: List[int], poles: List[NDArray[np.float64]],
                                     flat_poles: NDArray[np.float64], polygon_vertices: NDArray[np.float64]) -> NDArray[
        np.float64]:

        num_total_poles = flat_poles.shape[1]
        dist = np.zeros(num_total_poles)

        # Counter
        count_dist = 0

        for p in range(len(list_pole_vertices)):
            v = list_pole_vertices[p]
            num_poles = poles[p].shape[1]

            for j in range(num_poles):
                dist[count_dist] = np.linalg.norm(polygon_vertices[:, v] - poles[p][:, j])
                count_dist += 1

        return dist

    def vander_total(self, points: NDArray[np.float64], flat_poles: NDArray[np.float64],
                     distance_pole_vertex: NDArray[np.float64]) -> NDArray[np.float64]:

        """
        Compute rational function related to some polygon vertex, defined as (d_{v, k} / (z - z_{v,k})) where z_{v,k}
        is the kth pole related to vertex v and d_{v,k} is the distance from the kth pole and the vertex v itself

        z denotes complex number

        :param points: evaluation point (matrix 3 x num points)
        :param flat_poles: is a matrix 3 x num_total_poles containing the coordinates of poles
        :param distance_pole_vertex:
        :return: vander matrix: evaluation of rational functions at evaluation points (matrix num points x num rational functions)
        """

        assert points.shape[0] == 3

        num_quad_points = points.shape[1]
        num_total_poles = flat_poles.shape[1]
        vander_matrix = np.ones([num_quad_points, 2 * num_total_poles])

        vander_real_matrix = self.vander_real(points, flat_poles, distance_pole_vertex)
        vander_imag_matrix = self.vander_imag(points, flat_poles, distance_pole_vertex)

        for j in range(num_total_poles):
            vander_matrix[:, 2 * j] = vander_real_matrix[:, j]
            vander_matrix[:, 2 * j + 1] = vander_imag_matrix[:, j]

        return vander_matrix

    def vander_total_derivatives(self, points: NDArray[np.float64], flat_poles: NDArray[np.float64],
                                 distance_pole_vertex: NDArray[np.float64]):

        assert points.shape[0] == 3

        num_quad_points = points.shape[1]
        num_total_poles = flat_poles.shape[1]

        vandermonde_derivatives = np.zeros([2, num_quad_points, 2 * num_total_poles])

        vander_derivatives_real_matrix = self.vander_real_derivatives(points, flat_poles, distance_pole_vertex)
        vander_derivatives_imag_matrix = self.vander_imag_derivatives(points, flat_poles, distance_pole_vertex)

        for j in range(num_total_poles):
            vandermonde_derivatives[0, :, 2 * j] = vander_derivatives_real_matrix[0, :, j]
            vandermonde_derivatives[0, :, 2 * j + 1] = vander_derivatives_imag_matrix[0, :, j]

            vandermonde_derivatives[1, :, 2 * j] = vander_derivatives_real_matrix[1, :, j]
            vandermonde_derivatives[1, :, 2 * j + 1] = vander_derivatives_imag_matrix[1, :, j]

        return vandermonde_derivatives

    @staticmethod
    def vander_real(points: NDArray[np.float64], flat_poles: NDArray[np.float64],
                    distance_pole_vertex: NDArray[np.float64]) -> NDArray[np.float64]:

        assert points.shape[0] == 3

        diff = points[:, :, np.newaxis] - flat_poles[:, np.newaxis, :]  # Shape: (2, num_quad_points, num_total_poles)

        complex_module_sq = np.sum(diff ** 2, axis=0)  # Shape: (num_quad_points, num_total_poles)
        inv_complex_module = 1.0 / complex_module_sq

        vander_matrix = (diff[0, :, :] * inv_complex_module) * distance_pole_vertex[np.newaxis, :]

        return vander_matrix

    @staticmethod
    def vander_real_derivatives(points: NDArray[np.float64], flat_poles: NDArray[np.float64],
                                distance_pole_vertex: NDArray[np.float64]) -> NDArray[np.float64]:
        assert points.shape[0] == 3

        num_quad_points = points.shape[1]
        num_total_poles = flat_poles.shape[1]

        diff = points[:, :, np.newaxis] - flat_poles[:, np.newaxis, :]  # Shape: (2, num_quad_points, num_total_poles)

        complex_module_sq = np.sum(diff ** 2, axis=0)  # Shape: (num_quad_points, num_total_poles)
        inv_complex_module = 1.0 / complex_module_sq
        inv_complex_module_sq = inv_complex_module ** 2

        vandermonde_derivatives = np.zeros((2, num_quad_points, num_total_poles))
        vandermonde_derivatives[0, :, :] = (inv_complex_module - 2.0 * inv_complex_module_sq * diff[0, :,
                                                                                               :] ** 2) * distance_pole_vertex[
                                                                                                          np.newaxis, :]
        vandermonde_derivatives[1, :, :] = -2.0 * distance_pole_vertex[np.newaxis, :] * diff[0, :, :] * diff[1, :,
                                                                                                        :] * inv_complex_module_sq

        return vandermonde_derivatives

    @staticmethod
    def vander_imag(points: NDArray[np.float64], flat_poles: NDArray[np.float64],
                    distance_pole_vertex: NDArray[np.float64]) -> NDArray[np.float64]:

        assert points.shape[0] == 3

        diff = points[:, :, np.newaxis] - flat_poles[:, np.newaxis, :]  # Shape: (2, num_quad_points, num_total_poles)

        complex_module_sq = np.sum(diff ** 2, axis=0)  # Shape: (num_quad_points, num_total_poles)
        inv_complex_module = 1.0 / complex_module_sq

        vander_matrix = -(diff[1, :, :] * inv_complex_module) * distance_pole_vertex[np.newaxis, :]

        return vander_matrix

    @staticmethod
    def vander_imag_derivatives(points: NDArray[np.float64], flat_poles: NDArray[np.float64],
                                distance_pole_vertex: NDArray[np.float64]) -> NDArray[np.float64]:
        assert points.shape[0] == 3

        num_quad_points = points.shape[1]
        num_total_poles = flat_poles.shape[1]

        diff = points[:, :, np.newaxis] - flat_poles[:, np.newaxis, :]  # Shape: (2, num_quad_points, num_total_poles)

        complex_module_sq = np.sum(diff ** 2, axis=0)  # Shape: (num_quad_points, num_total_poles)
        inv_complex_module = 1.0 / complex_module_sq
        inv_complex_module_sq = inv_complex_module ** 2

        vandermonde_derivatives = np.zeros((2, num_quad_points, num_total_poles))
        vandermonde_derivatives[0, :, :] = 2.0 * distance_pole_vertex[np.newaxis, :] * diff[0, :, :] * diff[1, :,
                                                                                                       :] * inv_complex_module_sq
        vandermonde_derivatives[1, :, :] = (-inv_complex_module + 2.0 * inv_complex_module_sq * diff[1, :,
                                                                                                :] ** 2) * distance_pole_vertex[
                                                                                                           np.newaxis,
                                                                                                           :]

        return vandermonde_derivatives

    @staticmethod
    def num_rational_functions(num_rat_points: int, vander_type: RationalType) -> int:

        match vander_type:
            case RationalFunction.RationalType.total:
                return num_rat_points * 2
            case RationalFunction.RationalType.real:
                return num_rat_points
            case RationalFunction.RationalType.imag:
                return num_rat_points
            case _:
                raise ValueError('not valid rational type')

    def vander(self, points: NDArray[np.float64], flat_poles: NDArray[np.float64],
               distance_pole_vertex: NDArray[np.float64]) -> NDArray[np.float64]:

        match self.vander_type:
            case RationalFunction.RationalType.total:
                return self.vander_total(points, flat_poles, distance_pole_vertex)
            case RationalFunction.RationalType.real:
                return self.vander_real(points, flat_poles, distance_pole_vertex)
            case RationalFunction.RationalType.imag:
                return self.vander_imag(points, flat_poles, distance_pole_vertex)
            case _:
                raise ValueError('not valid rational type')

    def vander_derivatives(self, points: NDArray[np.float64], flat_poles: NDArray[np.float64],
                           distance_pole_vertex: NDArray[np.float64]) -> NDArray[np.float64]:

        match self.vander_type:
            case RationalFunction.RationalType.total:
                return self.vander_total_derivatives(points, flat_poles, distance_pole_vertex)
            case RationalFunction.RationalType.real:
                return self.vander_real_derivatives(points, flat_poles, distance_pole_vertex)
            case RationalFunction.RationalType.imag:
                return self.vander_imag_derivatives(points, flat_poles, distance_pole_vertex)
            case _:
                raise ValueError('not valid rational type')
