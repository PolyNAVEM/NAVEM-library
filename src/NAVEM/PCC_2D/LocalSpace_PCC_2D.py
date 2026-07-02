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
from typing import Dict, List, Tuple
from pypolydim import polydim
from NAVEM.PCC_2D import NAVEM_PCC_2D
from numpy.typing import NDArray
import numpy as np

from NAVEM.PCC_2D.NAVEM_Data_PCC_2D import NAVEMMappingType
from NAVEM.geometry.mesh_utilities import MeshGeometricData2D


class MethodTypes(Enum):
    NAVEM = 1
    FEM = 2
    VEM = 3

class ReferenceElementData:

    def __init__(self, method_type: MethodTypes, method_order: int):
        self.method_type: MethodTypes = method_type
        self.method_order: int = method_order

        match self.method_type:
            case MethodTypes.NAVEM:
                self.standard_method_type = None
                self.navem_reference_element_data: NAVEM_PCC_2D.NAVEMPCC2DReferenceElement = NAVEM_PCC_2D.NAVEMPCC2DReferenceElement(self.method_order)
            case MethodTypes.VEM:
                self.standard_method_type: polydim.pde_tools.local_space_pcc_2_d.MethodTypes = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.vem_pcc
                self.standard_reference_element_data: polydim.pde_tools.local_space_pcc_2_d.ReferenceElement_Data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(self.standard_method_type,
                                                                                                                                                                          method_order)
            case MethodTypes.FEM:
                self.standard_method_type: polydim.pde_tools.local_space_pcc_2_d.MethodTypes = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.fem_pcc
                self.standard_reference_element_data: polydim.pde_tools.local_space_pcc_2_d.ReferenceElement_Data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(self.standard_method_type,
                                                                                                                                                                          method_order)
            case _:
                raise ValueError("Not valid method type")

    def set_mesh_do_fs_info(self, mesh: gedim.MeshMatricesDAO,
                            boundary_info: Dict[int, polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo]) -> polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo:

        match self.method_type:
            case MethodTypes.NAVEM:

                num_cell0_ds = mesh.cell0_d_total_number()

                cells_num_do_fs_0_ds = []
                cells_boundary_info_0_ds = []
                for c in range(num_cell0_ds):
                    cells_num_do_fs_0_ds.append(
                        self.navem_reference_element_data.num_dof_0d)
                    cells_boundary_info_0_ds.append(
                        boundary_info[mesh.cell0_d_marker(c)])


                num_cell1_ds = mesh.cell1_d_total_number()

                cells_num_do_fs_1_ds = []
                cells_boundary_info_1_ds = []
                for c in range(num_cell1_ds):
                    cells_num_do_fs_1_ds.append(
                        self.navem_reference_element_data.num_dof_1d)
                    cells_boundary_info_1_ds.append(
                        boundary_info[mesh.cell1_d_marker(c)])

                num_cell2_ds = mesh.cell2_d_total_number()

                cells_num_do_fs_2_ds = []
                cells_boundary_info_2_ds = []
                for c in range(num_cell2_ds):
                    if mesh.cell2_d_number_vertices(c) == 3:
                        cells_num_do_fs_2_ds.append(
                            self.navem_reference_element_data.fem_reference_element_data.triangle_reference_element_data.num_dofs2_d)
                    else:
                        cells_num_do_fs_2_ds.append(
                            self.navem_reference_element_data.num_dof_2d)

                    cells_boundary_info_2_ds.append(
                        boundary_info[mesh.cell2_d_marker(c)])

                mesh_do_fs_info: polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo()
                mesh_do_fs_info.cells_num_do_fs = [cells_num_do_fs_0_ds, cells_num_do_fs_1_ds, cells_num_do_fs_2_ds, []]
                mesh_do_fs_info.cells_boundary_info = [cells_boundary_info_0_ds, cells_boundary_info_1_ds, cells_boundary_info_2_ds, []]

                return mesh_do_fs_info
            case MethodTypes.VEM:
                return polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(
                    self.standard_reference_element_data, mesh, boundary_info)
            case MethodTypes.FEM:
                return polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(
                    self.standard_reference_element_data, mesh, boundary_info)
            case _:
                raise ValueError("Not valid method type")

class LocalSpaceData:

    standard_local_space_data: polydim.pde_tools.local_space_pcc_2_d.LocalSpace_Data
    navem_local_space_data: NAVEM_PCC_2D.NAVEMPCC2DLocalSpace

    def __init__(self, geometry_utilities: gedim.GeometryUtilities,
                 mesh_geometric_data: MeshGeometricData2D,
                 reference_element_data: ReferenceElementData,
                 dictionary_file_name: str):


        match reference_element_data.method_type:
            case MethodTypes.NAVEM:
                self.navem_local_space_data \
                    = NAVEM_PCC_2D.NAVEMPCC2DLocalSpace(reference_element_data.navem_reference_element_data,
                                                        geometry_utilities)

                self.navem_local_space_data.setup_geometry(mesh_geometric_data)
                self.navem_local_space_data.setup_categories(dictionary_file_name)
                self.navem_local_space_data.setup_default_values(NAVEMMappingType.standard)
            case MethodTypes.VEM | MethodTypes.FEM:
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
            case MethodTypes.NAVEM:
                self.navem_local_space_data.create_local_space(cell_2_d_index,
                                                               reference_element_data.navem_reference_element_data,
                                                               NAVEMMappingType.standard)

            case MethodTypes.VEM | MethodTypes.FEM:
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
            case MethodTypes.NAVEM:
                return self.navem_local_space_data.basis_functions_values(reference_element_data.navem_reference_element_data, evaluation_points, evaluation_navem_input_output)
            case MethodTypes.VEM | MethodTypes.FEM:
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
            case MethodTypes.NAVEM:
                return self.navem_local_space_data.basis_functions_derivative_values(reference_element_data.navem_reference_element_data, evaluation_points, evaluation_navem_input_output)
            case MethodTypes.VEM | MethodTypes.FEM:
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

    def basis_functions_laplacian_values(self, reference_element_data: ReferenceElementData,
                                          projection_type: polydim.vem.pcc.ProjectionTypes = polydim.vem.pcc.ProjectionTypes.pi0km1_der,
                                          evaluation_points: NDArray[np.float64] = None,
                                          evaluation_navem_input_output = None) -> NDArray[np.float64]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM:
                return self.navem_local_space_data.basis_functions_laplacian_values(reference_element_data.navem_reference_element_data,
                                                                                    evaluation_points, evaluation_navem_input_output)
            case MethodTypes.VEM | MethodTypes.FEM:
                if evaluation_points is None:
                    return polydim.pde_tools.local_space_pcc_2_d.basis_functions_laplacian_values(
                        reference_element_data.standard_reference_element_data,
                        self.standard_local_space_data,
                        projection_type)
                else:
                    raise ValueError("not implemented method")
            case _:
                raise ValueError("Not valid method type")


    def internal_quadrature(self, reference_element_data: ReferenceElementData) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM:
                return self.navem_local_space_data.internal_quadrature()
            case MethodTypes.VEM | MethodTypes.FEM:
                internal_quadrature_data = polydim.pde_tools.local_space_pcc_2_d.internal_quadrature(reference_element_data.standard_reference_element_data,
                                                                                 self.standard_local_space_data)
                return internal_quadrature_data.points, internal_quadrature_data.weights
            case _:
                raise  ValueError("Not valid method type")

    def stabilization_matrix(self, reference_element_data: ReferenceElementData,
                             projection_type: polydim.vem.pcc.ProjectionTypes = polydim.vem.pcc.ProjectionTypes.pi_nabla) \
            -> NDArray[np.float64]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM:
                return np.zeros([self.navem_local_space_data.num_basis_functions, self.navem_local_space_data.num_basis_functions])
            case MethodTypes.VEM | MethodTypes.FEM:
                return polydim.pde_tools.local_space_pcc_2_d.stabilization_matrix(reference_element_data.standard_reference_element_data,
                                                                                  self.standard_local_space_data,
                                                                                  projection_type)
            case _:
                raise  ValueError("Not valid method type")

    def size(self, reference_element_data: ReferenceElementData) -> int:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM:
                return self.navem_local_space_data.num_basis_functions
            case MethodTypes.VEM | MethodTypes.FEM:
                return polydim.pde_tools.local_space_pcc_2_d.size(reference_element_data.standard_reference_element_data,
                                                                  self.standard_local_space_data)
            case _:
                raise  ValueError("Not valid method type")

    def edge_do_fs_coordinates(self, edge_local_index: int,
                              reference_element_data: ReferenceElementData) -> NDArray[np.float64]:

        match reference_element_data.method_type:
            case MethodTypes.NAVEM:
                return self.navem_local_space_data.edge_do_fs_coordinates(edge_local_index, reference_element_data.navem_reference_element_data)
            case MethodTypes.VEM | MethodTypes.FEM:
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
            case MethodTypes.NAVEM:
                return self.navem_local_space_data.basis_functions_values_on_edge(edge_local_index, reference_element_data.navem_reference_element_data, points_curvilinear_coordinates)
            case MethodTypes.VEM | MethodTypes.FEM:
                return polydim.pde_tools.local_space_pcc_2_d.basis_functions_values_on_edge(edge_local_index,
                                                                                            reference_element_data.standard_reference_element_data,
                                                                                            self.standard_local_space_data,
                                                                                            points_curvilinear_coordinates)
            case _:
                raise  ValueError("Not valid method type")


