from enum import Enum
from pypolydim import gedim
from typing import Dict, List
from pypolydim import polydim
from src.NAVEM.PCC_2D import NAVEM_PCC_2D
from numpy.typing import NDArray
import numpy as np
from src.GeDiM.geometry.geometry_utilities import MeshGeometricData2D


class MethodTypes(Enum):
    NAVEM = 1
    B_NAVEM = 2
    P_NAVEM = 3
    FEM_PCC = 4
    VEM_PCC = 5


class ReferenceElementData:

    navem_categories: Dict[int, NAVEM_PCC_2D.NNDictionary]

    def __init__(self, method_type: MethodTypes, method_order: int):
        self.method_type: MethodTypes = method_type
        self.method_order: int = method_order

        match self.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                self.standard_method_type: polydim.pde_tools.local_space_pcc_2_d.MethodTypes = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.vem_pcc
                self.standard_reference_element_data: polydim.pde_tools.local_space_pcc_2_d.ReferenceElement_Data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(self.standard_method_type,
                                                                                                                                                                                   method_order)

                self.fem_method_type: polydim.pde_tools.local_space_pcc_2_d.MethodTypes = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.fem_pcc
                self.fem_reference_element_data: polydim.pde_tools.local_space_pcc_2_d.ReferenceElement_Data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(
                    self.fem_method_type,
                    method_order)

                self.edge_internal_points = np.zeros(0)
                if self.standard_reference_element_data.vem_reference_element_data.quadrature.reference_edge_do_fs_internal_points.shape[1] > 0:
                    self.edge_internal_points = self.standard_reference_element_data.vem_reference_element_data.quadrature.reference_edge_do_fs_internal_points[0, :]

                self.utilities: polydim.vem.pcc.VEM_PCC_Utilities = polydim.vem.pcc.VEM_PCC_Utilities()
                self.edge_basis_coefficients = self.utilities.compute_edge_basis_coefficients(self.method_order, self.edge_internal_points)

            case MethodTypes.VEM_PCC:
                self.standard_method_type: polydim.pde_tools.local_space_pcc_2_d.MethodTypes = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.vem_pcc
                self.standard_reference_element_data: polydim.pde_tools.local_space_pcc_2_d.ReferenceElement_Data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(self.standard_method_type,
                                                                                                                                                                                   method_order)
            case MethodTypes.FEM_PCC:
                self.standard_method_type: polydim.pde_tools.local_space_pcc_2_d.MethodTypes = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.fem_pcc
                self.standard_reference_element_data: polydim.pde_tools.local_space_pcc_2_d.ReferenceElement_Data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(self.standard_method_type,
                                                                                                                                                                                   method_order)
            case _:
                raise ValueError("Not valid method type")

    def set_mesh_do_fs_info(self, mesh: gedim.MeshMatricesDAO,
                            boundary_info: Dict[int, polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo],
                            dictionary_file_name: str) -> polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo:

        match self.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:

                self.navem_categories = NAVEM_PCC_2D.categorize_elements_by_vertex_number(mesh, dictionary_file_name)


                num_cell0_ds = mesh.cell0_d_total_number()

                cells_num_do_fs_0_ds = []
                cells_boundary_info_0_ds = []
                for c in range(num_cell0_ds):
                    cells_num_do_fs_0_ds.append(
                        self.standard_reference_element_data.vem_reference_element_data.num_dofs0_d)
                    cells_boundary_info_0_ds.append(
                        boundary_info[mesh.cell0_d_marker(c)])


                num_cell1_ds = mesh.cell1_d_total_number()

                cells_num_do_fs_1_ds = []
                cells_boundary_info_1_ds = []
                for c in range(num_cell1_ds):
                    cells_num_do_fs_1_ds.append(
                        self.standard_reference_element_data.vem_reference_element_data.num_dofs1_d)
                    cells_boundary_info_1_ds.append(
                        boundary_info[mesh.cell1_d_marker(c)])

                num_cell2_ds = mesh.cell2_d_total_number()

                cells_num_do_fs_2_ds = []
                cells_boundary_info_2_ds = []
                for c in range(num_cell2_ds):
                    if mesh.cell2_d_number_vertices(c) == 3:
                        cells_num_do_fs_2_ds.append(
                            self.fem_reference_element_data.fem_reference_element_data.triangle_reference_element_data.num_dofs2_d)
                    else:
                        cells_num_do_fs_2_ds.append(
                            self.standard_reference_element_data.vem_reference_element_data.num_dofs2_d)

                    cells_boundary_info_2_ds.append(
                        boundary_info[mesh.cell2_d_marker(c)])

                mesh_do_fs_info: polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo()
                mesh_do_fs_info.cells_num_do_fs = [cells_num_do_fs_0_ds, cells_num_do_fs_1_ds, cells_num_do_fs_2_ds, []]
                mesh_do_fs_info.cells_boundary_info = [cells_boundary_info_0_ds, cells_boundary_info_1_ds, cells_boundary_info_2_ds, []]

                return mesh_do_fs_info
            case MethodTypes.VEM_PCC:
                return polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(
                    self.standard_reference_element_data, mesh, boundary_info)
            case MethodTypes.FEM_PCC:
                return polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(
                    self.standard_reference_element_data, mesh, boundary_info)
            case _:
                raise ValueError("Not valid method type")


class LocalSpaceData:

    cell_2_d_index: int
    num_vertices: int
    num_do_fs: int
    standard_local_space_data: polydim.pde_tools.local_space_pcc_2_d.LocalSpace_Data

    def __init__(self, geometry_utilities: gedim.GeometryUtilities,
                 mesh_geometric_data: MeshGeometricData2D,
                 reference_element_data: ReferenceElementData):

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                self.vem_geometry = polydim.vem.pcc.VEM_PCC_2D_Polygon_Geometry()

                quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()

                evaluation_points: Dict[int, NDArray[np.float64]] = {}
                quadrature_weights: Dict[int, NDArray[np.float64]] = {}
                for c in range(len(mesh_geometric_data.mesh_geometric_data.cell2_ds_vertices)):

                    internal_quadrature \
                        = quadrature.polygon_internal_quadrature(
                        reference_element_data.standard_reference_element_data.vem_reference_element_data.quadrature.reference_triangle_quadrature,
                        mesh_geometric_data.mesh_geometric_data.cell2_ds_triangulations[c])

                    evaluation_points[c] = internal_quadrature.points
                    quadrature_weights[c] = internal_quadrature.weights

                self.navem_input_output \
                    = NAVEM_PCC_2D.create_navem_input_output(geometry_utilities,
                                                             mesh_geometric_data,
                                                             NAVEM_PCC_2D.NAVEMType(reference_element_data.method_type.value),
                                                             reference_element_data.navem_categories,
                                                             evaluation_points,
                                                             quadrature_weights)

            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                pass
            case _:
                raise  ValueError("Not valid method type")


    def create_local_space(self,
                           tolerance_1_d: float,
                           tolerance_2_d: float,
                           mesh_geometric_data: MeshGeometricData2D,
                           cell_2_d_index: int,
                           reference_element_data: ReferenceElementData):

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                self.cell_2_d_index = cell_2_d_index
                self.num_vertices = mesh_geometric_data.mesh_geometric_data.cell2_ds_vertices[cell_2_d_index].shape[1]
                if self.num_vertices == 3:
                    self.standard_local_space_data: polydim.pde_tools.local_space_pcc_2_d.LocalSpace_Data \
                        = polydim.pde_tools.local_space_pcc_2_d.create_local_space(tolerance_1_d, tolerance_2_d,
                                                                                   mesh_geometric_data.mesh_geometric_data,
                                                                                   cell_2_d_index,
                                                                                   reference_element_data.fem_reference_element_data)

                    self.num_do_fs = self.standard_local_space_data.fem_local_space_data.number_of_basis_functions
                else:
                    self.num_do_fs = (reference_element_data.standard_reference_element_data.vem_reference_element_data.num_dofs2_d
                                      + reference_element_data.standard_reference_element_data.vem_reference_element_data.num_dofs1_d * self.num_vertices
                                      + reference_element_data.standard_reference_element_data.vem_reference_element_data.num_dofs0_d * self.num_vertices)

                    self.vem_geometry.tolerance1_d = tolerance_1_d
                    self.vem_geometry.tolerance1_d = tolerance_2_d
                    self.vem_geometry.vertices = mesh_geometric_data.mesh_geometric_data.cell2_ds_vertices[cell_2_d_index]
                    self.vem_geometry.centroid = mesh_geometric_data.mesh_geometric_data.cell2_ds_centroids[cell_2_d_index]
                    self.vem_geometry.measure = mesh_geometric_data.mesh_geometric_data.cell2_ds_areas[cell_2_d_index]
                    self.vem_geometry.diameter = mesh_geometric_data.mesh_geometric_data.cell2_ds_diameters[cell_2_d_index]
                    self.vem_geometry.triangulation_vertices = mesh_geometric_data.mesh_geometric_data.cell2_ds_triangulations[cell_2_d_index]
                    self.vem_geometry.edges_length = mesh_geometric_data.mesh_geometric_data.cell2_ds_edge_lengths[cell_2_d_index]
                    self.vem_geometry.edges_direction = mesh_geometric_data.mesh_geometric_data.cell2_ds_edge_directions[cell_2_d_index]
                    self.vem_geometry.edges_tangent = mesh_geometric_data.mesh_geometric_data.cell2_ds_edge_tangents[cell_2_d_index]
                    self.vem_geometry.edges_normal = mesh_geometric_data.mesh_geometric_data.cell2_ds_edge_normals[cell_2_d_index]

            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                self.standard_local_space_data: polydim.pde_tools.local_space_pcc_2_d.LocalSpace_Data \
                    = polydim.pde_tools.local_space_pcc_2_d.create_local_space(tolerance_1_d, tolerance_2_d,
                                                                               mesh_geometric_data.mesh_geometric_data,
                                                                               cell_2_d_index,
                                                                               reference_element_data.standard_reference_element_data)
            case _:
                raise  ValueError("Not valid method type")

    def basis_functions_values(self, reference_element_data: ReferenceElementData,
                               projection_type: polydim.vem.pcc.ProjectionTypes = polydim.vem.pcc.ProjectionTypes.pi0k,
                               evaluation_points: NDArray[np.float64] = None,
                               evaluation_navem_input_output = None) -> NDArray[np.float64]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                if self.num_vertices == 3:
                    if evaluation_points is None:
                        return polydim.pde_tools.local_space_pcc_2_d.basis_functions_values(
                            reference_element_data.fem_reference_element_data,
                            self.standard_local_space_data,
                            projection_type)
                    else:
                        return polydim.pde_tools.local_space_pcc_2_d.basis_functions_values(
                            reference_element_data.fem_reference_element_data,
                            self.standard_local_space_data,
                            evaluation_points,
                            projection_type)
                else:
                    if evaluation_points is None:
                        return self.navem_input_output[self.cell_2_d_index].basis_values
                    else:
                        return evaluation_navem_input_output[self.cell_2_d_index].basis_values
            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                if evaluation_points is None:
                    return polydim.pde_tools.local_space_pcc_2_d.basis_functions_values(reference_element_data.standard_reference_element_data,
                                                                                        self.standard_local_space_data,
                                                                                        projection_type)
                else:
                    return polydim.pde_tools.local_space_pcc_2_d.basis_functions_values(reference_element_data.standard_reference_element_data,
                                                                                        self.standard_local_space_data,
                                                                                        evaluation_points,
                                                                                        projection_type)
            case _:
                raise  ValueError("Not valid method type")


    def basis_functions_derivative_values(self, reference_element_data: ReferenceElementData,
                                          projection_type: polydim.vem.pcc.ProjectionTypes = polydim.vem.pcc.ProjectionTypes.pi0km1_der,
                                          evaluation_points: NDArray[np.float64] = None,
                                          evaluation_navem_input_output = None) -> List[NDArray[np.float64]]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                if self.num_vertices == 3:
                    if evaluation_points is None:
                        return polydim.pde_tools.local_space_pcc_2_d.basis_functions_derivative_values(
                            reference_element_data.fem_reference_element_data,
                            self.standard_local_space_data,
                            projection_type)
                    else:
                        return polydim.pde_tools.local_space_pcc_2_d.basis_functions_derivative_values(
                            reference_element_data.fem_reference_element_data,
                            self.standard_local_space_data,
                            evaluation_points,
                            projection_type)
                else:
                    if evaluation_points is None:
                        return self.navem_input_output[self.cell_2_d_index].basis_derivatives_values
                    else:
                        return evaluation_navem_input_output[self.cell_2_d_index].basis_derivatives_values
            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                if evaluation_points is None:
                    return polydim.pde_tools.local_space_pcc_2_d.basis_functions_derivative_values(
                        reference_element_data.standard_reference_element_data,
                        self.standard_local_space_data,
                        projection_type)
                else:
                    return polydim.pde_tools.local_space_pcc_2_d.basis_functions_derivative_values(
                        reference_element_data.standard_reference_element_data,
                        self.standard_local_space_data,
                        evaluation_points,
                        projection_type)
            case _:
                raise ValueError("Not valid method type")


    def internal_quadrature(self, reference_element_data: ReferenceElementData) -> gedim.quadrature.QuadratureData:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                if self.num_vertices == 3:
                    return polydim.pde_tools.local_space_pcc_2_d.internal_quadrature(reference_element_data.fem_reference_element_data,
                                                                                     self.standard_local_space_data)
                else:

                    assert len(self.navem_input_output[self.cell_2_d_index].internal_quadrature.weights) != 0

                    return self.navem_input_output[self.cell_2_d_index].internal_quadrature
            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                return polydim.pde_tools.local_space_pcc_2_d.internal_quadrature(reference_element_data.standard_reference_element_data,
                                                                                 self.standard_local_space_data)
            case _:
                raise  ValueError("Not valid method type")

    def stabilization_matrix(self, reference_element_data: ReferenceElementData, projection_type: polydim.vem.pcc.ProjectionTypes = polydim.vem.pcc.ProjectionTypes.pi_nabla) -> NDArray[np.float64]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                return np.zeros([self.num_do_fs, self.num_do_fs])
            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                return polydim.pde_tools.local_space_pcc_2_d.stabilization_matrix(reference_element_data.standard_reference_element_data,
                                                                                  self.standard_local_space_data,
                                                                                  projection_type)
            case _:
                raise  ValueError("Not valid method type")

    def size(self, reference_element_data: ReferenceElementData) -> int:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                return self.num_do_fs
            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                return polydim.pde_tools.local_space_pcc_2_d.size(reference_element_data.standard_reference_element_data,
                                                                  self.standard_local_space_data)
            case _:
                raise  ValueError("Not valid method type")

    def edge_do_fs_coordinates(self, edge_local_index: int,
                              reference_element_data: ReferenceElementData) -> NDArray[np.float64]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                if self.num_vertices == 3:
                    return polydim.pde_tools.local_space_pcc_2_d.edge_dofs_coordinates(
                    reference_element_data.fem_reference_element_data,
                    self.standard_local_space_data,
                    edge_local_index)
                else:
                    reference_edge_do_fs_point = reference_element_data.standard_reference_element_data.vem_reference_element_data.Quadrature.ReferenceEdgeDOFsInternalPoints
                    num_edge_do_fs = reference_edge_do_fs_point.shape[1]

                    if num_edge_do_fs == 0:
                        return np.zeros([0, 0])

                    num_edges = self.vem_geometry.vertices.shape[1]
                    edge_origin = self.vem_geometry.vertices[:, edge_local_index] if self.vem_geometry.edges_direction[edge_local_index] else self.vem_geometry.vertices[:, (edge_local_index + 1) % num_edges]
                    edge_tangent = self.vem_geometry.edges_tangent[:, edge_local_index]
                    edge_direction = 1.0 if self.vem_geometry.edges_direction[edge_local_index] else -1.0

                    edge_do_fs_coordinates = np.zeros([3, num_edge_do_fs])
                    for r in range(num_edge_do_fs):
                        edge_do_fs_coordinates[:, r] = edge_origin + edge_direction * reference_edge_do_fs_point[0, r] * edge_tangent
                    return edge_do_fs_coordinates
            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                return polydim.pde_tools.local_space_pcc_2_d.edge_dofs_coordinates(
                    reference_element_data.standard_reference_element_data,
                    self.standard_local_space_data,
                    edge_local_index)
            case _:
                raise  ValueError("Not valid method type")

    def basis_functions_values_on_edge(self, edge_local_index: int,
                                       reference_element_data: ReferenceElementData,
                                       points_curvilinear_coordinates: NDArray[np.float64]) -> NDArray[np.float64]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM | MethodTypes.B_NAVEM | MethodTypes.P_NAVEM:
                if self.num_vertices == 3:
                    return polydim.pde_tools.local_space_pcc_2_d.basis_functions_values_on_edge(edge_local_index,
                                                                                                reference_element_data.standard_reference_element_data,
                                                                                                self.standard_local_space_data,
                                                                                                points_curvilinear_coordinates)
                else:


                    return reference_element_data.utilities.compute_values_on_edge(reference_element_data.edge_internal_points,
                                                                                   reference_element_data.method_order,
                                                                                   reference_element_data.edge_basis_coefficients,
                                                                                   points_curvilinear_coordinates)
            case MethodTypes.VEM_PCC | MethodTypes.FEM_PCC:
                return polydim.pde_tools.local_space_pcc_2_d.basis_functions_values_on_edge(edge_local_index,
                                                                                            reference_element_data.standard_reference_element_data,
                                                                                            self.standard_local_space_data,
                                                                                            points_curvilinear_coordinates)
            case _:
                raise  ValueError("Not valid method type")


