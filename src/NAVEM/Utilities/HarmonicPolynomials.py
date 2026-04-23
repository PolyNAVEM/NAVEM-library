import numpy as np
from numpy.typing import NDArray
from enum import Enum
from typing import Tuple


class HarmonicPolynomials:

    class HarmonicType(Enum):
        total = 1
        real = 2
        imag = 3

    def __init__(self, degree: int, vander_type: HarmonicType = HarmonicType.total):

        assert degree >= 0

        self.dimension = 2
        self.vander_type = vander_type
        self.degree = degree
        self.num_total_harmonic_polynomials = 1 + degree * 2
        self.num_real_harmonic_polynomials = 1 + degree
        self.num_imag_harmonic_polynomials = 1 + degree

        match self.vander_type:
            case HarmonicPolynomials.HarmonicType.total:
                self.num_harmonic_polynomials = self.num_total_harmonic_polynomials
            case HarmonicPolynomials.HarmonicType.real:
                self.num_harmonic_polynomials = self.num_real_harmonic_polynomials
            case HarmonicPolynomials.HarmonicType.imag:
                self.num_harmonic_polynomials = self.num_imag_harmonic_polynomials
            case _:
                raise ValueError('not valid harmonic type')

    def vander_total(self, points: NDArray[np.float64], diameter: float, centroid: NDArray[np.float64]) -> NDArray[np.float64]:

        vander_matrix = np.ones([points.shape[1], self.num_total_harmonic_polynomials])

        if self.degree == 0:
            return vander_matrix


        inv_diameter: float = 1.0 / diameter
        x = (points[0, :] - centroid[0]) * inv_diameter
        y = (points[1, :] - centroid[1]) * inv_diameter

        count_harmonic_polynomials: int = 0

        vander_matrix[:, 1] = x
        vander_matrix[:, 2] = y
        count_harmonic_polynomials += 2

        for _ in range(2, self.degree + 1):

            x_prev = vander_matrix[:, count_harmonic_polynomials - 1]
            y_prev = vander_matrix[:, count_harmonic_polynomials]

            vander_matrix[:, count_harmonic_polynomials + 1] = x * x_prev - y * y_prev
            vander_matrix[:, count_harmonic_polynomials + 2] = y * x_prev + x * y_prev
            count_harmonic_polynomials += 2

        return vander_matrix

    def vander_total_derivatives(self, vander_matrix: NDArray[np.float64], diameter: float) -> NDArray[np.float64]:
        num_quadrature = vander_matrix.shape[0]
        vandermonde_derivatives = np.zeros([2, num_quadrature, self.num_total_harmonic_polynomials])

        if self.degree == 0:
            return vandermonde_derivatives

        inv_diameter: float = 1.0 / diameter

        vandermonde_derivatives[0, :, 1] = inv_diameter * vander_matrix[:, 0]
        vandermonde_derivatives[1, :, 2] = inv_diameter * vander_matrix[:, 0]

        count_harmonic_polynomials: int = 2

        for p in range(2, self.degree + 1):
            x_prev = p * inv_diameter * vander_matrix[:, count_harmonic_polynomials - 1]
            y_prev = p * inv_diameter * vander_matrix[:, count_harmonic_polynomials]

            vandermonde_derivatives[0, :, count_harmonic_polynomials + 1] = x_prev
            vandermonde_derivatives[1, :, count_harmonic_polynomials + 1] = - y_prev

            vandermonde_derivatives[0, :, count_harmonic_polynomials + 2] = y_prev
            vandermonde_derivatives[1, :, count_harmonic_polynomials + 2] = x_prev

            count_harmonic_polynomials += 2

        return vandermonde_derivatives

    def vander_real(self, vander_total_matrix) -> NDArray[np.float64]:

        vander_matrix = np.ones([vander_total_matrix.shape[0], self.num_real_harmonic_polynomials])

        for d in range(self.num_real_harmonic_polynomials):

            if d == 0:
                vander_matrix[:, d] = vander_total_matrix[:, d]
            else:
                vander_matrix[:, d] = vander_total_matrix[:, 2 * d - 1]

        return vander_matrix

    def vander_real_derivatives(self, vander_matrix: NDArray[np.float64], diameter: float) -> NDArray[np.float64]:
        num_quadrature = vander_matrix.shape[0]
        vandermonde_derivatives = np.zeros([2, num_quadrature, self.num_real_harmonic_polynomials])

        if self.degree == 0:
            return vandermonde_derivatives

        inv_diameter: float = 1.0 / diameter

        vandermonde_derivatives[0, :, 1] = inv_diameter * vander_matrix[:, 0]

        id_harmonic_polynomials = 1
        count_harmonic_polynomials = 2

        for p in range(2, self.degree + 1):
            vandermonde_derivatives[0, :, id_harmonic_polynomials + 1] = p * inv_diameter * vander_matrix[:, count_harmonic_polynomials - 1]
            vandermonde_derivatives[1, :, id_harmonic_polynomials + 1] = - p * inv_diameter * vander_matrix[:, count_harmonic_polynomials]

            id_harmonic_polynomials += 1
            count_harmonic_polynomials += 2

        return vandermonde_derivatives

    def vander_imag(self, vander_total_matrix: NDArray[np.float64]) -> NDArray[np.float64]:

        vander_matrix = np.ones([vander_total_matrix.shape[0], self.num_imag_harmonic_polynomials])

        for d in range(self.num_imag_harmonic_polynomials):

            if d == 0:
                vander_matrix[:, d] = vander_total_matrix[:, d]
            else:
                vander_matrix[:, d] = vander_total_matrix[:, 2 * d]

        return vander_matrix

    def vander_imag_derivatives(self, vander_matrix: NDArray[np.float64], diameter: float) -> NDArray[np.float64]:
        num_quadrature = vander_matrix.shape[0]
        vandermonde_derivatives = np.zeros([2, num_quadrature, self.num_imag_harmonic_polynomials])

        if self.degree != 0:
            inv_diameter: float = 1.0 / diameter
            count_harmonic_polynomials: int = 0
            id_harmonic_polynomials: int = 0

            if self.degree >= 1:
                vandermonde_derivatives[0, :, id_harmonic_polynomials + 1] = np.zeros(num_quadrature)
                vandermonde_derivatives[1, :, id_harmonic_polynomials + 1] = inv_diameter * vander_matrix[:, 0]

                id_harmonic_polynomials += 1
                count_harmonic_polynomials += 2

            if self.degree >= 2:
                for p in np.arange(2, self.degree + 1):
                    vandermonde_derivatives[0, :, id_harmonic_polynomials + 1] = p * inv_diameter * vander_matrix[:, count_harmonic_polynomials]
                    vandermonde_derivatives[1, :, id_harmonic_polynomials + 1] = p * inv_diameter * vander_matrix[:, count_harmonic_polynomials - 1]

                    id_harmonic_polynomials += 1
                    count_harmonic_polynomials += 2

        return vandermonde_derivatives

    def vander(self, points: NDArray[np.float64], diameter: float, centroid: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        vander_total_matrix = self.vander_total(points, diameter, centroid)

        match self.vander_type:
            case HarmonicPolynomials.HarmonicType.total:
                return vander_total_matrix, vander_total_matrix
            case HarmonicPolynomials.HarmonicType.real:
                return self.vander_real(vander_total_matrix), vander_total_matrix
            case HarmonicPolynomials.HarmonicType.imag:
                return self.vander_imag(vander_total_matrix), vander_total_matrix
            case _:
                raise ValueError('not valid harmonic type')

    def vander_derivatives(self, vander_total_matrix: NDArray[np.float64], diameter: float) -> NDArray[np.float64]:

        match self.vander_type:
            case HarmonicPolynomials.HarmonicType.total:
                return self.vander_total_derivatives(vander_total_matrix, diameter)
            case HarmonicPolynomials.HarmonicType.real:
                return self.vander_real_derivatives(vander_total_matrix, diameter)
            case HarmonicPolynomials.HarmonicType.imag:
                return self.vander_imag_derivatives(vander_total_matrix, diameter)
            case _:
                raise ValueError('not valid harmonic type')
