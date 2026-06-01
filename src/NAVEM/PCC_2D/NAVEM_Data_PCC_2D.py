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

from enum import Enum
from pypolydim import gedim
from typing import List
import numpy as np
from numpy.typing import NDArray


class NAVEMType(Enum):
    H_NAVEM = 1
    B_NAVEM = 2
    P_NAVEM = 3

class NAVEMMappingType(Enum):
    standard = 1
    inertia = 2
    rotated_inertia = 3

class NAVEMElementType(Enum):
    generic_convex = 1
    generic_concave = 2

class BasisFunctionType(Enum):
    vertex = 1
    edge = 2
    internal = 3

class NAVEMOutput:

    def __init__(self):
        self.basis_values: NDArray[np.float64] = np.zeros((0, 0))
        self.basis_derivatives_values: List[NDArray[np.float64]] = [np.zeros((0, 0)) for _ in range(2)]
        self.basis_laplacian_values: NDArray[np.float64] = np.zeros((0, 0))
        self.internal_quadrature: gedim.quadrature.QuadratureData = gedim.quadrature.QuadratureData()
        self.internal_quadrature_per_do_fs: List[gedim.quadrature.QuadratureData] = []

class NNDictionary:

    def __init__(self):
        self.list_elements: List[int] = []
        self.method_order: int = 1
        self.method_type: NAVEMType = NAVEMType.H_NAVEM
        self.neural_network = {BasisFunctionType.vertex: None, BasisFunctionType.edge: None, BasisFunctionType.internal: None}

class NNCategory:

    tolerance_hanging_nodes = 0.05

    def __init__(self, num_vertices: int, element_type: NAVEMElementType):
        self.num_vertices: int = num_vertices
        self.element_type: NAVEMElementType = element_type


    def __eq__(self, other):
        if not isinstance(other, NNCategory):
            return NotImplemented
        return (
            self.num_vertices == other.num_vertices and
            self.element_type == other.element_type
        )

    def __hash__(self):
        return hash((self.num_vertices, self.element_type))

    @staticmethod
    def is_concave(internal_angles: List[float]):
        return any(a > np.pi * (1.0 + NNCategory.tolerance_hanging_nodes) for a in internal_angles)