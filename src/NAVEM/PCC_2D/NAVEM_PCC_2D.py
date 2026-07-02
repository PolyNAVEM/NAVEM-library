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

from NAVEM.NeuralNetwork import h_navem_network, b_navem_network, p_navem_network
from NAVEM.NeuralNetwork import exact_bc_navem_network_utilities
from NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import SetupDerivatives, AbstractBPNAVEM
from NAVEM.Utilities import NAVEMGenerators, RationalFunction
import tensorflow as tf
from NAVEM.geometry.mesh_utilities import MeshGeometricData2D
from NAVEM.PCC_2D.NAVEM_Data_PCC_2D import *
from typing import Dict, Tuple
from pypolydim import polydim
from scipy.linalg import cho_factor, cho_solve


class NAVEMPCC2DReferenceElement:

    def __init__(self, method_order: int):

        self.dimension = 2

        self.method_order = method_order
        self.num_dof_2d = int(method_order * (method_order - 1) * 0.5)
        self.num_dof_1d = method_order - 1
        self.num_dof_0d = 1

        fem_reference_element: polydim.fem.pcc.FEM_PCC_2D_ReferenceElement = polydim.fem.pcc.FEM_PCC_2D_ReferenceElement()
        self.fem_reference_element_data: polydim.fem.pcc.FEM_PCC_2D_ReferenceElement_Data = fem_reference_element.create(method_order)


class NAVEMPCC2DLocalSpace:

    cell_2_d_index: int
    num_vertices: int

    # Polygon data
    fem_polygon: polydim.fem.pcc.FEM_PCC_2D_Polygon_Geometry
    fem_local_space_data: polydim.fem.pcc.FEM_PCC_2D_LocalSpace_Data

    num_boundary_basis_functions: int
    num_basis_functions: int

    vander_boundary: NDArray[np.float64]
    vander_internal: NDArray[np.float64]
    h_matrix: NDArray[np.float64]

    boundary_quadrature_data: polydim.vem.quadrature.VEM_Quadrature_2D.Edges_QuadratureData

    d_matrix: NDArray[np.float64]
    laplacian_coefficients: NDArray[np.float64]

    mesh_geometric_data: MeshGeometricData2D
    polygon_vertices: NDArray[np.float64]
    polygon_diameter: float
    polygon_measure: float
    polygon_centroid: NDArray[np.float64]
    polygon_edges_directions: List[bool]
    polygon_edges_tangents: NDArray[np.float64]
    polygon_edges_lengths: List[float]
    polygon_edge_normals: NDArray[np.float64]
    polygon_triangulations: List[NDArray[np.float64]]


    navem_categories: Dict[NNCategory, NNDictionary]
    internal_quadrature_points: Dict[int, NDArray[np.float64]]
    internal_quadrature_weights: Dict[int, NDArray[np.float64]]
    navem_input_output: Dict[int, NAVEMOutput]
    navem_element_type: NAVEMMappingType

    def __init__(self, reference_element_data: NAVEMPCC2DReferenceElement,
                 geometry_utilities: gedim.GeometryUtilities,
                 monomial_scaling: float = 0.5):

        self.utilities = polydim.vem.pcc.VEM_PCC_Utilities()

        self.geometry_utilities = geometry_utilities
        self.scaling = monomial_scaling
        self.method_order = reference_element_data.method_order

        self.monomials = polydim.utilities.Monomials_2D()
        self.monomials_data = self.monomials.compute(self.method_order)

        self.n_k = int(self.monomials_data.num_monomials)
        self.num_internal_basis_functions = 0 if self.method_order == 1 \
            else int(self.method_order * (self.method_order - 1) * 0.5)

        self.quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
        self.reference_quadrature_data: polydim.vem.quadrature.VEM_QuadratureData_2D = self.quadrature.compute_pcc_2_d(self.method_order)

        self.reference_triangle_quadrature: gedim.quadrature.QuadratureData = gedim.quadrature.Quadrature_Gauss2D_Triangle.fill_points_and_weights(2)

        self.edge_internal_points = np.zeros(0)
        if self.reference_quadrature_data.reference_edge_do_fs_internal_points.shape[1] > 0:
            self.edge_internal_points = self.reference_quadrature_data.reference_edge_do_fs_internal_points[0, :]

        self.utilities: polydim.vem.pcc.VEM_PCC_Utilities = polydim.vem.pcc.VEM_PCC_Utilities()
        self.edge_basis_coefficients = self.utilities.compute_edge_basis_coefficients(self.method_order,
                                                                                      self.edge_internal_points)

        self.fem_local_space: polydim.fem.pcc.FEM_PCC_2D_LocalSpace = polydim.fem.pcc.FEM_PCC_2D_LocalSpace()



    def setup_geometry(self, mesh_geometric_data: MeshGeometricData2D):
        self.mesh_geometric_data = mesh_geometric_data

    def setup_categories(self, dictionary_file_name: str):
        self.navem_categories = categorize_elements_by_vertex_number(self.method_order,
                                                                     self.mesh_geometric_data,
                                                                     dictionary_file_name)

    def setup_default_values(self, navem_element_type):
        self.internal_quadrature_points: Dict[int, NDArray[np.float64]] = {}
        self.internal_quadrature_weights: Dict[int, NDArray[np.float64]] = {}
        self.navem_element_type = navem_element_type
        for c in range(len(self.mesh_geometric_data.cell2_ds_vertices)):
            internal_quadrature \
                = self.quadrature.polygon_internal_quadrature(
                self.reference_triangle_quadrature,
                self.mesh_geometric_data.cell2_ds_triangulations[c])

            self.internal_quadrature_points[c] = internal_quadrature.points
            self.internal_quadrature_weights[c] = internal_quadrature.weights


        self.navem_input_output \
            = create_navem_input_output(self.geometry_utilities,
                                        self.mesh_geometric_data,
                                        self.navem_categories,
                                        self.internal_quadrature_points,
                                        self.internal_quadrature_weights,
                                        navem_element_type=self.navem_element_type)

    def create_local_space(self,
                           cell_index: int,
                           reference_element_data: NAVEMPCC2DReferenceElement,
                           navem_element_type: NAVEMMappingType) -> None:

        self.cell_2_d_index = cell_index
        self.num_vertices = self.mesh_geometric_data.cell2_ds_vertices[cell_index].shape[1]

        if self.num_vertices == 3:
            self.fem_polygon = polydim.fem.pcc.FEM_PCC_2D_Polygon_Geometry()
            self.fem_polygon.tolerance2_d = self.geometry_utilities.tolerance1_d()
            self.fem_polygon.tolerance1_d = self.geometry_utilities.tolerance2_d()
            self.fem_polygon.vertices = self.mesh_geometric_data.cell2_ds_vertices[cell_index]
            self.fem_polygon.edges_direction = self.mesh_geometric_data.cell2_ds_edge_directions[cell_index]
            self.fem_polygon.edges_tangent = self.mesh_geometric_data.cell2_ds_edge_tangents[cell_index]
            self.fem_polygon.edges_length = self.mesh_geometric_data.cell2_ds_edge_lengths[cell_index]
            self.fem_local_space_data: polydim.fem.pcc.FEM_PCC_2D_LocalSpace_Data \
                = self.fem_local_space.create_local_space(reference_element_data.fem_reference_element_data, self.fem_polygon)

            self.num_basis_functions = self.fem_local_space_data.number_of_basis_functions
        else:
            match navem_element_type:
                case NAVEMMappingType.standard:
                    self.polygon_vertices = self.mesh_geometric_data.cell2_ds_vertices[cell_index]
                    self.polygon_diameter = self.mesh_geometric_data.cell2_ds_diameters[cell_index]
                    self.polygon_centroid = self.mesh_geometric_data.cell2_ds_centroids[cell_index]
                    self.polygon_measure = self.mesh_geometric_data.cell2_ds_areas[cell_index]
                    self.polygon_edges_directions = self.mesh_geometric_data.cell2_ds_edge_directions[cell_index]
                    self.polygon_edges_tangents = self.mesh_geometric_data.cell2_ds_edge_tangents[cell_index]
                    self.polygon_edges_lengths = self.mesh_geometric_data.cell2_ds_edge_lengths[cell_index]
                    self.polygon_edge_normals = self.mesh_geometric_data.cell2_ds_edge_normals[cell_index]
                    self.polygon_triangulations = self.mesh_geometric_data.cell2_ds_triangulations[cell_index]
                case NAVEMMappingType.inertia:
                    self.polygon_vertices = self.mesh_geometric_data.cell2_ds_inertia_vertices[cell_index]
                    self.polygon_diameter = self.mesh_geometric_data.cell2_ds_inertia_diameters[cell_index]
                    self.polygon_measure = self.mesh_geometric_data.cell2_ds_inertia_areas[cell_index]
                    self.polygon_centroid = self.mesh_geometric_data.cell2_ds_inertia_centroids[cell_index]
                    self.polygon_edges_directions = self.mesh_geometric_data.cell2_ds_edge_directions[cell_index]
                    self.polygon_edges_tangents = self.mesh_geometric_data.cell2_ds_inertia_edges_tangents[cell_index]
                    self.polygon_edges_lengths = self.mesh_geometric_data.cell2_ds_inertia_edges_lengths[cell_index]
                    self.polygon_edge_normals = self.mesh_geometric_data.cell2_ds_inertia_edges_normals[cell_index]
                    self.polygon_triangulations = self.mesh_geometric_data.cell2_ds_inertia_triangulations[cell_index]
                case _:
                    raise ValueError("Invalid navem element type")


            self.num_boundary_basis_functions = self.method_order * self.polygon_vertices.shape[1]
            self.num_basis_functions = self.num_boundary_basis_functions + self.num_internal_basis_functions

            internal_quadrature_data = self.quadrature.polygon_internal_quadrature(
                self.reference_quadrature_data.reference_triangle_quadrature,
                self.polygon_triangulations)

            self.boundary_quadrature_data = self.quadrature.polygon_edges_lobatto_quadrature(self.reference_quadrature_data.reference_edge_do_fs_internal_points,
                                                                                   self.reference_quadrature_data.reference_edge_do_fs_internal_weights,
                                                                                   self.reference_quadrature_data.reference_edge_do_fs_extrema_weights,
                                                                                   self.polygon_vertices,
                                                                                   self.polygon_edges_lengths,
                                                                                   self.polygon_edges_directions,
                                                                                   self.polygon_edges_tangents,
                                                                                   self.polygon_edge_normals)

            self.vander_boundary = self.monomials.vander(self.monomials_data,
                                               self.boundary_quadrature_data.quadrature.points,
                                               self.polygon_centroid,
                                               self.scaling * self.polygon_diameter)

            self.vander_internal = self.monomials.vander(self.monomials_data,
                                                         internal_quadrature_data.points,
                                                         self.polygon_centroid,
                                                         self.scaling * self.polygon_diameter)

            self.h_matrix = (self.vander_internal.T
                             @ np.diag(internal_quadrature_data.weights)
                             @ self.vander_internal)

            self.laplacian_coefficients = self.compute_laplacian_coefficients()

    def basis_functions_values(self, reference_element_data: NAVEMPCC2DReferenceElement,
                               evaluation_points: NDArray[np.float64] = None,
                               evaluation_navem_input_output = None) -> NDArray[np.float64]:

        if self.num_vertices == 3:
            if evaluation_points is None:
                return self.fem_local_space.compute_basis_functions_values(
                    reference_element_data.fem_reference_element_data,
                    self.fem_local_space_data)
            else:
                return self.fem_local_space.compute_basis_functions_values(
                    reference_element_data.fem_reference_element_data,
                    self.fem_local_space_data,
                    evaluation_points)
        else:
            if evaluation_points is None:
                return self.navem_input_output[self.cell_2_d_index].basis_values
            else:
                return evaluation_navem_input_output[self.cell_2_d_index].basis_values

    def basis_functions_derivative_values(self, reference_element_data: NAVEMPCC2DReferenceElement,
                                          evaluation_points: NDArray[np.float64] = None,
                                          evaluation_navem_input_output = None) -> List[NDArray[np.float64]]:

        if self.num_vertices == 3:
            if evaluation_points is None:
                return self.fem_local_space.compute_basis_functions_derivative_values(
                    reference_element_data.fem_reference_element_data,
                    self.fem_local_space_data)
            else:
                return self.fem_local_space.compute_basis_functions_derivative_values(
                    reference_element_data.fem_reference_element_data,
                    self.fem_local_space_data,
                    evaluation_points)
        else:
            if evaluation_points is None:
                return self.navem_input_output[self.cell_2_d_index].basis_derivatives_values
            else:
                return evaluation_navem_input_output[self.cell_2_d_index].basis_derivatives_values

    def basis_functions_laplacian_values(self, reference_element_data: NAVEMPCC2DReferenceElement,
                                          evaluation_points: NDArray[np.float64] = None,
                                          _evaluation_navem_input_output = None) -> NDArray[np.float64]:

        if self.num_vertices == 3:
            if evaluation_points is None:
                return self.fem_local_space.compute_basis_functions_laplacian_values(
                    reference_element_data.fem_reference_element_data,
                    self.fem_local_space_data)
            else:
                return self.fem_local_space.compute_basis_functions_laplacian_values(
                    reference_element_data.fem_reference_element_data,
                    self.fem_local_space_data,
                    evaluation_points)
        else:
            raise ValueError("not implemented")

    def internal_quadrature(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

            if self.num_vertices == 3:
                return self.fem_local_space_data.internal_quadrature.points, self.fem_local_space_data.internal_quadrature.weights
            else:
                return self.internal_quadrature_points[self.cell_2_d_index], self.internal_quadrature_weights[self.cell_2_d_index]


    def edge_do_fs_coordinates(self, edge_local_index: int,
                              reference_element_data: NAVEMPCC2DReferenceElement) -> NDArray[np.float64]:

        if self.num_vertices == 3:
            return self.fem_local_space.edge_do_fs_coordinates(
            reference_element_data.fem_reference_element_data,
            self.fem_local_space_data,
            edge_local_index)
        else:
            reference_edge_do_fs_point = self.reference_quadrature_data.reference_edge_do_fs_internal_points
            num_edge_do_fs = reference_edge_do_fs_point.shape[1]

            if num_edge_do_fs == 0:
                return np.zeros([0, 0])

            num_edges = self.polygon_vertices.shape[1]
            edge_origin = self.polygon_vertices[:, edge_local_index] if self.polygon_edges_directions[edge_local_index] else self.polygon_vertices[:, (edge_local_index + 1) % num_edges]
            edge_tangent = self.polygon_edges_tangents[:, edge_local_index]
            edge_direction = 1.0 if self.polygon_edges_directions[edge_local_index] else -1.0

            edge_do_fs_coordinates = np.zeros([3, num_edge_do_fs])
            for r in range(num_edge_do_fs):
                edge_do_fs_coordinates[:, r] = edge_origin + edge_direction * reference_edge_do_fs_point[0, r] * edge_tangent
            return edge_do_fs_coordinates


    def basis_functions_values_on_edge(self, _edge_local_index: int,
                                       reference_element_data: NAVEMPCC2DReferenceElement,
                                       points_curvilinear_coordinates: NDArray[np.float64]) -> NDArray[np.float64]:

        if self.num_vertices == 3:
            return self.fem_local_space.compute_basis_functions_values_on_edge(reference_element_data.fem_reference_element_data, self.fem_local_space_data, points_curvilinear_coordinates)
        else:
            return self.utilities.compute_values_on_edge(self.edge_internal_points,
                                                         reference_element_data.method_order,
                                                         self.edge_basis_coefficients,
                                                         points_curvilinear_coordinates)


    def compute_standard_polynomial_do_fs(self) -> NDArray[np.float64]:

        d_matrix = np.zeros((self.num_basis_functions, self.n_k))

        d_matrix[:self.num_boundary_basis_functions, :] = self.vander_boundary

        if self.method_order > 1:

            d_matrix[self.num_boundary_basis_functions:, :] \
                = ((1.0 / (self.scaling * self.scaling * self.polygon_measure))
                   * self.h_matrix[0:self.num_internal_basis_functions, :])

        return d_matrix.T

    def compute_polynomial_do_fs(self) -> NDArray[np.float64]:


        d_matrix = np.zeros((self.num_basis_functions, self.n_k))

        d_matrix[:self.num_boundary_basis_functions, :] = self.vander_boundary

        if self.method_order > 1:

            d_matrix[self.num_boundary_basis_functions:, :] \
                = ((1.0 / (self.scaling * self.scaling * self.polygon_measure))
                   * (self.monomials_data.laplacian[:, 0:self.num_internal_basis_functions]
                   @self. h_matrix[0:self.num_internal_basis_functions, 0:self.num_internal_basis_functions]).T)

        return d_matrix.T

    def compute_laplacian_coefficients(self) -> NDArray[np.float64]:
        """
        This function use internal DOFs that are moments of laplacian.
        """

        rhs = np.eye(self.num_internal_basis_functions)
        lhs = self.h_matrix[:self.num_internal_basis_functions, :self.num_internal_basis_functions]

        return cho_solve(cho_factor(lhs), rhs)

    def evaluate_laplacian_internal_basis_functions(self, points: NDArray[np.float64])\
            -> NDArray[np.float64]:


        vander_matrix = self.monomials.vander(self.monomials_data,
                                              points,
                                              self.polygon_centroid,
                                              self.scaling * self.polygon_diameter)

        return vander_matrix[:, :self.num_internal_basis_functions] @ self.laplacian_coefficients

    def convert_do_fs_id_internal_basis_functions_to_binary(self):

        """
        create all possible combinations of 0s and 1s
        """
        if self.method_order == 1:
            return np.zeros(0, dtype=np.int8)
        elif self.method_order == 2:
            return np.zeros([0, 0], dtype=np.int8)

        max_n = self.num_internal_basis_functions - 1

        arr = np.arange(max_n + 1)
        m = int(np.log2(max_n) + 1)
        to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
        strs = to_str_func(arr)
        binary_do_fs = np.zeros(list(arr.shape) + [m], dtype=np.int8)
        for bit_ix in range(0, m):
            fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
            binary_do_fs[..., bit_ix] = fetch_bit_func(strs).astype("int8")

        return binary_do_fs

    def define_inputs_functions_ids(self, basis_function_type):
        id_basis_functions = None
        match basis_function_type:

            case BasisFunctionType.vertex:
                pass
            case BasisFunctionType.edge:

                if self.method_order == 1:
                    raise ValueError("not valid basis function type for this order")

                if self.method_order > 2:
                    id_basis_functions = np.expand_dims(np.arange(self.method_order - 1) + 1, axis=1)
                else:
                    id_basis_functions = np.zeros(0)

            case BasisFunctionType.internal:

                if self.method_order == 1:
                    raise ValueError("not valid basis function type for this order")

                id_basis_functions = self.convert_do_fs_id_internal_basis_functions_to_binary()
            case _:
                raise ValueError("not valid basis function type")

        return id_basis_functions


def categorize_elements_by_vertex_number(method_order: int,
                                         mesh_geometric_data: MeshGeometricData2D,
                                         dictionary_file_path: str) -> Dict[NNCategory, NNDictionary]:

    categories: Dict[NNCategory, NNDictionary] = {}
    with open(dictionary_file_path + "/dictionary.txt", "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            element_class, name_storage = line.split("=", 1)
            num_vertices, element_type, order, basis_function_type_value = element_class.split(",", 4)
            num_vertices = int(num_vertices.strip())
            element_type = NAVEMElementType(int(element_type.strip()))
            order = int(order.strip())
            basis_function_type_value = int(basis_function_type_value.strip())
            category = NNCategory(num_vertices, element_type)

            name_storage = name_storage.strip()
            name_storage = dictionary_file_path + "/" + name_storage

            if category not in categories:
                categories[category] = NNDictionary()

            raw: Dict[str, str] = {}
            with open(name_storage + "/dictionary.txt", "r") as dictionary_file:
                for dictionary_line in dictionary_file:
                    dictionary_line = dictionary_line.strip()
                    if not dictionary_line:
                        continue

                    key, value = dictionary_line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    raw[key] = value

            if method_order != order:
                continue

            categories[category].method_type = NAVEMType(int(raw["method_type"]))

            assert basis_function_type_value == int(raw["basis_function_type"])
            assert order == int(raw["method_order"])
            assert num_vertices == int(raw["num_vertices"])
            assert element_type.value == int(raw["element_type"])

            basis_function_type = BasisFunctionType(basis_function_type_value)

            match categories[category].method_type:
                case NAVEMType.H_NAVEM:
                    flags = h_navem_network.load_flags_from_dictionary(name_storage, raw)
                    categories[category].neural_network[basis_function_type] = h_navem_network.HNAVEMNetworksContainer(flags)
                    categories[category].neural_network[basis_function_type].load_weights(name_storage)
                case NAVEMType.B_NAVEM:
                    flags = exact_bc_navem_network_utilities.load_flags_from_dictionary(name_storage, raw)
                    categories[category].neural_network[basis_function_type] = b_navem_network.BNAVEMNetwork(flags)
                    categories[category].neural_network[basis_function_type].load_model(name_storage)
                case NAVEMType.P_NAVEM:
                    flags = exact_bc_navem_network_utilities.load_flags_from_dictionary(name_storage, raw)
                    categories[category].neural_network[basis_function_type] = p_navem_network.PNAVEMNetwork(flags)
                    categories[category].neural_network[basis_function_type].load_model(name_storage)
                case _:
                    raise ValueError("Unknown method type")

    for c in range(len(mesh_geometric_data.cell2_ds_vertices)):

        num_vertices = mesh_geometric_data.cell2_ds_vertices[c].shape[1]

        if num_vertices == 3:
            continue

        internal_angles = mesh_geometric_data.cell2_ds_inertia_internal_angles[c]

        sum_internal_angles = sum(internal_angles)

        if abs(sum_internal_angles - (num_vertices - 2) * np.pi) > np.pi * NNCategory.tolerance_hanging_nodes:
            raise ValueError("polygon is not simple")

        is_concave = NNCategory.is_concave(internal_angles)

        if is_concave:
            element_type = NAVEMElementType.generic_concave
        else:
            element_type = NAVEMElementType.generic_convex

        element_class = NNCategory(num_vertices, element_type)

        if element_class not in categories:
            raise ValueError("Not valid element")

        categories[element_class].list_elements.append(c)

    return categories


def navem_predict_basis_values_and_derivatives(geometry_utilities: gedim.GeometryUtilities,
                                               mesh_geometric_data: MeshGeometricData2D,
                                               neural_network: Dict[BasisFunctionType, h_navem_network.HNAVEMNetworksContainer],
                                               evaluation_points: Dict[int, NDArray[np.float64]],
                                               evaluation_weights: Dict[int, NDArray[np.float64]],
                                               predict_laplacian: bool = False,
                                               navem_element_type: NAVEMMappingType = NAVEMMappingType.standard) -> Dict[int, NAVEMOutput]:

    if predict_laplacian:
        raise ValueError("not implemented method")

    navem_generators = NAVEMGenerators.NAVEMGenerators(geometry_utilities,
                                                       neural_network[BasisFunctionType.vertex].flags["num_vertices"],
                                                       neural_network[BasisFunctionType.vertex].flags["harmonic_degree"],
                                                       neural_network[BasisFunctionType.vertex].flags["use_hanging_function"],
                                                       neural_network[BasisFunctionType.vertex].flags["normalization_diameter"],
                                                       neural_network[BasisFunctionType.vertex].flags["num_rationals_points"],
                                                       RationalFunction.RationalFunction.RationalType(neural_network[BasisFunctionType.vertex].flags["rational_type_function"]),
                                                       NAVEMElementType(neural_network[BasisFunctionType.vertex].flags["element_type"]))

    num_vertices = neural_network[BasisFunctionType.vertex].flags["num_vertices"]
    network_input_dimension = 2 + 2 * (num_vertices - 1)

    inputs = np.array([], dtype=np.float64).reshape(0, network_input_dimension - 2)
    num_generators = neural_network[BasisFunctionType.vertex].flags["num_generators"]

    n_elements = len(evaluation_points)
    global_jac_inv = np.zeros([n_elements * num_vertices, 2, 2])

    n_local_pts = 0
    for c, points in evaluation_points.items():
        n_local_pts = points.shape[1]
        break

    super_vander = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))
    super_vander_dx = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))
    super_vander_dy = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))

    output_values: Dict[int, NAVEMOutput] = {}
    c_pol = 0
    for c, internal_nodes in evaluation_points.items():

        polygon = mesh_geometric_data.cell2_ds_polygon[c]
        mapped_angles = np.expand_dims(np.array(mesh_geometric_data.cell2_ds_inertia_internal_angles[c]), axis=1).T

        output_values[c] = NAVEMOutput()

        match navem_element_type:
            case NAVEMMappingType.standard:
                output_values[c].internal_quadrature.points = internal_nodes
                output_values[c].internal_quadrature.weights = evaluation_weights[c]
            case _:
                raise ValueError("not valid navem element type")

        for v_id in range(num_vertices):

            # Create input
            mapped_vertices, mapped_pts, jac_inv, scaling = polygon.map_f_inv(internal_nodes, v_id)
            mapped_vertices = np.roll(mapped_vertices, axis=1, shift=-v_id)
            internal_angles = np.roll(mapped_angles, shift=-v_id)
            vertex_distance = scaling * np.roll(polygon.mapped_max_vertex_distance, shift=-v_id)

            match navem_element_type:
                case NAVEMMappingType.standard:
                    global_jac_inv[c_pol * num_vertices + v_id, :, :] = jac_inv[:2, :2]
                case _:
                    raise ValueError("not valid navem element type")

            c_inputs = np.expand_dims(mapped_vertices[:2, 1:].flatten(), 0)

            inputs = np.concatenate([inputs, c_inputs])

            # Store the vandermonde matrix that will be used
            local_vandermonde, local_vandermonde_grads = (
                navem_generators.vander_and_vander_derivatives(mapped_pts, mapped_vertices,
                                                               internal_angles=internal_angles,
                                                               vertex_distance=vertex_distance))

            super_vander[c_pol * num_vertices + v_id, :, :] = local_vandermonde
            super_vander_dx[c_pol * num_vertices + v_id, :, :] = local_vandermonde_grads[0, :, :]
            super_vander_dy[c_pol * num_vertices + v_id, :, :] = local_vandermonde_grads[1, :, :]

        c_pol += 1


    inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)

    super_vander = tf.convert_to_tensor(super_vander, dtype=tf.float64)
    super_vander_dx = tf.convert_to_tensor(super_vander_dx, dtype=tf.float64)
    super_vander_dy = tf.convert_to_tensor(super_vander_dy, dtype=tf.float64)

    basis_coefficients = neural_network[BasisFunctionType.vertex].nn_basis_function.call(inputs)
    basis_values = neural_network[BasisFunctionType.vertex].nn_basis_function.apply_vandermonde(basis_coefficients, super_vander)
    basis_values = tf.reshape(basis_values, [-1]).numpy()

    derivatives_coefficients = neural_network[BasisFunctionType.vertex].nn_basis_derivatives.call(inputs)
    dx_values = neural_network[BasisFunctionType.vertex].nn_basis_derivatives.apply_vandermonde(derivatives_coefficients, super_vander_dx)
    dy_values = neural_network[BasisFunctionType.vertex].nn_basis_derivatives.apply_vandermonde(derivatives_coefficients, super_vander_dy)
    dx_values = tf.reshape(dx_values, [-1]).numpy()
    dy_values = tf.reshape(dy_values, [-1]).numpy()



    offset_point = 0
    offset_func = 0
    for c, _ in evaluation_points.items():

        output_values[c].basis_derivatives_values[0] = np.zeros([n_local_pts, num_vertices])
        output_values[c].basis_derivatives_values[1] = np.zeros([n_local_pts, num_vertices])
        output_values[c].basis_values = np.zeros([n_local_pts, num_vertices])

        output_values[c].basis_laplacian_values = None

        for i in range(num_vertices):
            matrix_jac_inv_transpose = global_jac_inv[offset_func, 0:2, 0:2].T
            output_values[c].basis_derivatives_values[0][:, i] \
                = (matrix_jac_inv_transpose[0, 0] * dx_values[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[0, 1] * dy_values[offset_point: offset_point + n_local_pts])
            output_values[c].basis_derivatives_values[1][:, i]  \
                = (matrix_jac_inv_transpose[1, 0] * dx_values[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[1, 1] * dy_values[offset_point: offset_point + n_local_pts])

            output_values[c].basis_values[:, i] \
                = basis_values[offset_point:offset_point + n_local_pts]

            offset_func += 1
            offset_point += n_local_pts

    return output_values


def exact_bc_navem_predict_basis_values_and_derivatives(geometry_utilities: gedim.GeometryUtilities,
                                                        mesh_geometric_data: MeshGeometricData2D,
                                                        neural_network: Dict[BasisFunctionType, AbstractBPNAVEM],
                                                        evaluation_points: Dict[int, NDArray[np.float64]],
                                                        evaluation_weights: Dict[int, NDArray[np.float64]],
                                                        predict_laplacian: bool = False,
                                                        navem_element_type: NAVEMMappingType = NAVEMMappingType.standard) -> Dict[int, NAVEMOutput]:

    num_vertices = neural_network[BasisFunctionType.vertex].flags["num_vertices"]
    network_input_dimension = 2 * num_vertices

    n_elements = len(evaluation_points)
    global_jac_inv = np.zeros([n_elements * num_vertices, 2, 2])

    n_local_pts = 0
    for c, points in evaluation_points.items():
        n_local_pts = points.shape[1]
        break

    offset_func = 0
    offset_points = 0

    c_pol = 0

    xy_per_pol = np.zeros(shape=(n_elements, n_local_pts, 2))
    vertices_per_pol = np.zeros(shape=(n_elements, 2, num_vertices))
    jac_per_pol = np.zeros(shape=(n_elements, 2, 2, num_vertices))
    inputs = np.zeros(shape=(n_elements * n_local_pts * num_vertices, network_input_dimension))

    output_values: Dict[int, NAVEMOutput] = {}
    for c, internal_nodes in evaluation_points.items():

        output_values[c] = NAVEMOutput()

        polygon = mesh_geometric_data.cell2_ds_polygon[c]

        inertia_mapped_internal_nodes, jac_inv_inertia = polygon.map_inertia_inv(internal_nodes)
        xy_per_pol[c_pol, :, :] = inertia_mapped_internal_nodes[:2, :].T
        vertices_per_pol[c_pol, :, :] = polygon.mapped_vertices[:2, :]

        match navem_element_type:
            case NAVEMMappingType.standard:
                output_values[c].internal_quadrature.points = internal_nodes
                output_values[c].internal_quadrature.weights = evaluation_weights[c]
            case NAVEMMappingType.inertia:
                output_values[c].internal_quadrature.points = inertia_mapped_internal_nodes
                output_values[c].internal_quadrature.weights = (evaluation_weights[c]
                                                                * geometry_utilities.polygon_area(mesh_geometric_data.cell2_ds_polygon[c].mapped_vertices)
                                                                / mesh_geometric_data.cell2_ds_areas[c])
            case NAVEMMappingType.rotated_inertia:
                output_values[c].internal_quadrature_per_do_fs = [gedim.quadrature.QuadratureData() for _ in range(num_vertices)]
            case _:
                raise ValueError("not valid navem element type")

        for v_id in range(num_vertices):
            # Create input
            rotated_vertices, rotated_mapped_points, jac_inv, jac, _ = polygon.map_fix_vertex_inv(
                polygon.mapped_vertices,
                inertia_mapped_internal_nodes,
                v_id)

            mapped_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)

            match navem_element_type:
                case NAVEMMappingType.standard:
                    global_jac_inv[offset_func, :, :] = jac_inv[:2, :2] @ jac_inv_inertia[:2, :2]
                case NAVEMMappingType.inertia:
                    global_jac_inv[offset_func, :, :] = jac_inv[:2, :2]
                case NAVEMMappingType.rotated_inertia:
                    global_jac_inv[offset_func, :, :] = np.eye(2)
                    output_values[c].internal_quadrature_per_do_fs[v_id].points = rotated_mapped_points
                    output_values[c].internal_quadrature_per_do_fs[v_id].weights = (evaluation_weights[c]
                                                                * abs(geometry_utilities.polygon_area(rotated_vertices))
                                                                / mesh_geometric_data.cell2_ds_areas[c])
                case _:
                    raise ValueError("not valid navem element type")

            flat_rotated_vertices = mapped_vertices[:2, 1:].flatten()
            rep_flat_rot_vertices = np.tile(flat_rotated_vertices[np.newaxis, :], [n_local_pts, 1])
            c_inputs = np.concatenate([rotated_mapped_points[:2, :].T, rep_flat_rot_vertices], axis=1)

            inputs[offset_func * n_local_pts: (offset_func + 1) * n_local_pts, :] = c_inputs
            jac_per_pol[c_pol, :, :, v_id] = jac[:2, :2]
            offset_func += 1

        c_pol += 1
        offset_points += 1

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)

    du_ref_dxx = None
    du_ref_dxy = None
    du_ref_dyy = None
    if predict_laplacian:
        neural_network[BasisFunctionType.vertex].setup_model_global_input(xy_per_pol, vertices_per_pol, jac_per_pol, SetupDerivatives.basis_and_derivatives_and_laplacian, geometry_utilities)
        laplacian_output = neural_network[BasisFunctionType.vertex].get_second_derivatives_u(inputs).numpy()
        du_ref_dxx = laplacian_output[:, 0]
        du_ref_dxy = laplacian_output[:, 1]
        du_ref_dyy = laplacian_output[:, 2]
    else:
        neural_network[BasisFunctionType.vertex].setup_model_global_input(xy_per_pol, vertices_per_pol, jac_per_pol, SetupDerivatives.basis_and_derivatives, geometry_utilities)

    net_output = neural_network[BasisFunctionType.vertex].get_u_and_du(inputs).numpy()

    u = net_output[:, 0]
    du_ref_dx = net_output[:, 1]
    du_ref_dy = net_output[:, 2]

    offset_point = 0
    offset_func = 0
    for c, _ in evaluation_points.items():

        output_values[c].basis_derivatives_values[0] = np.zeros([n_local_pts, num_vertices])
        output_values[c].basis_derivatives_values[1] = np.zeros([n_local_pts, num_vertices])
        output_values[c].basis_values = np.zeros([n_local_pts, num_vertices])

        if predict_laplacian:
            output_values[c].basis_laplacian_values = np.zeros([n_local_pts, num_vertices])
        else:
            output_values[c].basis_laplacian_values = None

        for i in range(num_vertices):
            matrix_jac_inv_transpose = global_jac_inv[offset_func, 0:2, 0:2].T
            output_values[c].basis_derivatives_values[0][:, i] \
                = (matrix_jac_inv_transpose[0, 0] * du_ref_dx[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[0, 1] * du_ref_dy[offset_point: offset_point + n_local_pts])
            output_values[c].basis_derivatives_values[1][:, i] \
                = (matrix_jac_inv_transpose[1, 0] * du_ref_dx[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[1, 1] * du_ref_dy[offset_point: offset_point + n_local_pts])

            output_values[c].basis_values[:, i]\
                = u[offset_point:offset_point + n_local_pts]

            if predict_laplacian:
                b_lap = matrix_jac_inv_transpose @ matrix_jac_inv_transpose.T
                output_values[c].basis_laplacian_values[:, i] = (b_lap[0, 0] * du_ref_dxx[offset_point: offset_point + n_local_pts]
                                                                 + b_lap[0, 1] * du_ref_dxy[offset_point: offset_point + n_local_pts]
                                                                 + b_lap[1, 0] * du_ref_dxy[offset_point: offset_point + n_local_pts]
                                                                 + b_lap[1, 1] * du_ref_dyy[offset_point: offset_point + n_local_pts])

            offset_func += 1
            offset_point += n_local_pts

    return output_values

def reproduce_polynomials(_polygon_vertices: NDArray[np.float64],
                          basis_function_values: NDArray[np.float64],
                          basis_function_derivatives_values: List[NDArray[np.float64]],
                          basis_laplacian_values: NDArray[np.float64] = None) \
        -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], NDArray[np.float64]]:

    basis_function_values[:, -1] = 1.0 - np.sum(basis_function_values[:, 0:-1], axis=1)
    basis_function_derivatives_values[0][:, -1] = - np.sum(basis_function_derivatives_values[0][:, 0:-1],
                                                           axis=1)
    basis_function_derivatives_values[1][:, -1] = - np.sum(basis_function_derivatives_values[1][:, 0:-1],
                                                           axis=1)

    if basis_laplacian_values is not None:
        basis_laplacian_values[:, -1] = - np.sum(basis_laplacian_values[:, 0:-1], axis=1)

    return basis_function_values, basis_function_derivatives_values, basis_laplacian_values

def create_navem_input_output(geometry_utilities: gedim.GeometryUtilities,
                              mesh_geometric_data: MeshGeometricData2D,
                              navem_categories: Dict[NNCategory, NNDictionary],
                              evaluation_points: Dict[int, NDArray[np.float64]],
                              evaluation_weights: Dict[int, NDArray[np.float64]] = None,
                              predict_laplacian: bool = False,
                              navem_element_type: NAVEMMappingType = NAVEMMappingType.standard) -> Dict[int, NAVEMOutput]:

        outputs: Dict[int, NAVEMOutput] = {}

        for element_class, dictionary in navem_categories.items():

            if len(dictionary.list_elements) == 0:
                continue

            list_points: Dict[int, NDArray[np.float64]] = {}
            list_weights: Dict[int, NDArray[np.float64]] = {}

            for c in dictionary.list_elements:

                if c not in evaluation_points:
                    continue

                outputs[c] = NAVEMOutput()
                list_points[c] = evaluation_points[c]
                list_weights[c] = evaluation_weights[c] if evaluation_weights is not None else np.zeros([0])

            dictionary_outputs: Dict[int, NAVEMOutput] = {}
            match dictionary.method_type:
                case NAVEMType.H_NAVEM:
                    dictionary_outputs \
                        = navem_predict_basis_values_and_derivatives(geometry_utilities,
                                                                     mesh_geometric_data,
                                                                     dictionary.neural_network,
                                                                     list_points,
                                                                     evaluation_weights=list_weights,
                                                                     predict_laplacian=predict_laplacian,
                                                                     navem_element_type=navem_element_type)
                case NAVEMType.B_NAVEM | NAVEMType.P_NAVEM:
                    dictionary_outputs \
                        = exact_bc_navem_predict_basis_values_and_derivatives(geometry_utilities,
                                                                              mesh_geometric_data,
                                                                              dictionary.neural_network,
                                                                              list_points,
                                                                              evaluation_weights=list_weights,
                                                                              predict_laplacian=predict_laplacian,
                                                                              navem_element_type=navem_element_type)
                case _:
                    raise ValueError("Not valid method type")

            for c in dictionary.list_elements:
                outputs[c].basis_values, outputs[c].basis_derivatives_values, outputs[c].basis_laplacian_values \
                    = reproduce_polynomials(
                    mesh_geometric_data.cell2_ds_vertices[c],
                    dictionary_outputs[c].basis_values,
                    dictionary_outputs[c].basis_derivatives_values,
                    dictionary_outputs[c].basis_laplacian_values)

                outputs[c].internal_quadrature = dictionary_outputs[c].internal_quadrature
                outputs[c].internal_quadrature_per_do_fs = dictionary_outputs[c].internal_quadrature_per_do_fs

        return outputs
