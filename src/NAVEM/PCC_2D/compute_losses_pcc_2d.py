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
from NAVEM.NeuralNetwork.h_navem_network import BoundaryLoss
from NAVEM.PCC_2D import NAVEM_PCC_2D, LocalSpace_PCC_2D
from pypolydim import gedim, polydim
from NAVEM.geometry.mesh_utilities import MeshGeometricData2D
from typing import Dict, Tuple
from numpy.typing import NDArray


def compute_boundary_loss(geometry_utilities: gedim.GeometryUtilities,
                          mesh_geometric_data: MeshGeometricData2D,
                          reference_element_data: LocalSpace_PCC_2D.ReferenceElementData,
                          local_space_data: LocalSpace_PCC_2D.LocalSpaceData) -> Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:

    assert reference_element_data.method_order == 1

    num_cell_2 = len(mesh_geometric_data.cell2_ds_vertices)

    test_l2_loss_cells = np.zeros([num_cell_2])
    test_h1_loss_cells = np.zeros([num_cell_2])

    quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
    reference_quadrature = gedim.quadrature.Quadrature_Gauss1D()
    reference_quadrature_data = reference_quadrature.fill_points_and_weights(20)

    evaluation_points: Dict[int, NDArray[np.float64]] = {}
    evaluation_weights: Dict[int, NDArray[np.float64]] = {}
    for c in range(num_cell_2):

        vertices = mesh_geometric_data.cell2_ds_vertices[c]
        quadrature_data = quadrature.polygon_edges_quadrature(reference_quadrature_data,
                                                              vertices,
                                                              mesh_geometric_data.cell2_ds_edge_lengths[c],
                                                              [True for _ in range(vertices.shape[1])],
                                                              mesh_geometric_data.cell2_ds_edge_tangents[c],
                                                              mesh_geometric_data.cell2_ds_edge_normals[c])

        evaluation_points[c] = quadrature_data.quadrature.points
        evaluation_weights[c] = quadrature_data.quadrature.weights


    evaluation_navem_input_output = None
    match reference_element_data.method_type:
        case LocalSpace_PCC_2D.MethodTypes.NAVEM:
            evaluation_navem_input_output = NAVEM_PCC_2D.create_navem_input_output(geometry_utilities,
                                                                                   mesh_geometric_data,
                                                                                   reference_element_data.navem_categories,
                                                                                   evaluation_points,
                                                                                   navem_element_type=NAVEM_PCC_2D.NAVEMMappingType.standard)
        case _:
            pass

    boundary_loss = BoundaryLoss(geometry_utilities, method_order=reference_element_data.method_order)
    denominator = 0
    for c in range(num_cell_2):

        local_space_data.create_local_space(geometry_utilities.tolerance1_d(),
                                            geometry_utilities.tolerance2_d(),
                                            mesh_geometric_data,
                                            c,
                                            reference_element_data)

        basis_functions_values = local_space_data.basis_functions_values(reference_element_data,
                                                                         polydim.vem.pcc.ProjectionTypes.pi0k,
                                                                         evaluation_points[c],
                                                                         evaluation_navem_input_output)

        basis_functions_derivatives_values \
            = local_space_data.basis_functions_derivative_values(reference_element_data,
                                                                 polydim.vem.pcc.ProjectionTypes.pi0km1_der,
                                                                 evaluation_points[c],
                                                                 evaluation_navem_input_output)

        polygon_vertices = mesh_geometric_data.cell2_ds_vertices[c]
        edge_tangents = mesh_geometric_data.cell2_ds_edge_tangents[c]
        edge_lengths = mesh_geometric_data.cell2_ds_edge_lengths[c]
        num_vertices = polygon_vertices.shape[1]

        labels = np.zeros([evaluation_points[c].shape[1], num_vertices])
        labels_tangent_derivatives = np.zeros([evaluation_points[c].shape[1], num_vertices])
        basis_tangent_derivatives = np.zeros([evaluation_points[c].shape[1], num_vertices])
        for i in range(num_vertices):
            boundary_loss.initialize_basis_evaluations_on_boundary(reference_quadrature_data.points[0, :],
                                                                   num_vertices, v_id=i)
            _, labels[:, i], _, labels_tangent_derivatives[:, i] = boundary_loss.add_polygon(polygon_vertices)

            num_points_on_each_edge = reference_quadrature_data.points.shape[1]
            for e in range(num_vertices):
                basis_tangent_derivatives[e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] \
                    = (basis_functions_derivatives_values[0][e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] * edge_tangents[0, e]
                       + basis_functions_derivatives_values[1][e * num_points_on_each_edge: (e + 1) * num_points_on_each_edge, i] * edge_tangents[1, e]) / edge_lengths[e]

        test_l2_loss_cells[c] = np.sum((evaluation_weights[c] @
                                        (basis_functions_values - labels) ** 2),axis=0)

        test_h1_loss_cells[c] = np.sum((evaluation_weights[c] @
                                        (basis_tangent_derivatives - labels_tangent_derivatives) ** 2),axis=0)

        denominator += basis_functions_values.shape[1]

    test_l2_loss = np.sqrt(np.sum(test_l2_loss_cells) / denominator)
    test_h1_loss = np.sqrt(np.sum(test_h1_loss_cells) / denominator)

    return test_l2_loss, test_h1_loss, test_l2_loss_cells, test_h1_loss_cells

def compute_polynomial_loss(geometry_utilities: gedim.GeometryUtilities,
                            mesh_geometric_data: MeshGeometricData2D,
                            reference_element_data: LocalSpace_PCC_2D.ReferenceElementData,
                            local_space_data: LocalSpace_PCC_2D.LocalSpaceData) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

    num_cell_2 = len(mesh_geometric_data.cell2_ds_vertices)

    monomials = polydim.utilities.Monomials_2D()
    monomials_data = monomials.compute(reference_element_data.method_order)

    test_l2_loss_cells = np.zeros([num_cell_2, monomials_data.num_monomials])
    test_h1_loss_cells = np.zeros([num_cell_2, monomials_data.num_monomials])

    quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
    reference_quadrature = gedim.quadrature.Quadrature_Gauss2D_Triangle()
    reference_quadrature_data = reference_quadrature.fill_points_and_weights(20)
    evaluation_points: Dict[int, NDArray[np.float64]] = {}
    evaluation_weights: Dict[int, NDArray[np.float64]] = {}
    for c in range(num_cell_2):
        internal_quadrature = quadrature.polygon_internal_quadrature(reference_quadrature_data,
                                                                     mesh_geometric_data.cell2_ds_triangulations[
                                                                         c])

        evaluation_points[c] = internal_quadrature.points
        evaluation_weights[c] = internal_quadrature.weights

        assert geometry_utilities.is_value_zero(abs(np.sum(internal_quadrature.weights) - mesh_geometric_data.cell2_ds_areas[c]), geometry_utilities.tolerance2_d())

    evaluation_navem_input_output = None
    match reference_element_data.method_type:
        case LocalSpace_PCC_2D.MethodTypes.NAVEM:
            evaluation_navem_input_output = NAVEM_PCC_2D.create_navem_input_output(geometry_utilities,
                                                                                   mesh_geometric_data,
                                                                                   reference_element_data.navem_categories,
                                                                                   evaluation_points,
                                                                                   navem_element_type=NAVEM_PCC_2D.NAVEMMappingType.standard)
        case _:
            pass

    for c in range(num_cell_2):
        local_space_data.create_local_space(geometry_utilities.tolerance1_d(),
                                            geometry_utilities.tolerance2_d(),
                                            mesh_geometric_data,
                                            c,
                                            reference_element_data)

        polygon_vertices = mesh_geometric_data.cell2_ds_vertices[c]
        basis_functions_values = local_space_data.basis_functions_values(reference_element_data,
                                                                         polydim.vem.pcc.ProjectionTypes.pi0k,
                                                                         evaluation_points[c],
                                                                         evaluation_navem_input_output)

        basis_functions_derivatives_values \
            = local_space_data.basis_functions_derivative_values(reference_element_data,
                                                                 polydim.vem.pcc.ProjectionTypes.pi0km1_der,
                                                                 evaluation_points[c],
                                                                 evaluation_navem_input_output)

        monomials_values = monomials.vander(monomials_data, evaluation_points[c], np.zeros([3]), diam=1.0)
        monomials_derivatives_values = monomials.vander_derivatives(monomials_data, monomials_values, diam=1.0)

        monomials_do_fs = monomials.vander(monomials_data, polygon_vertices, np.zeros([3]), diam=1.0)

        test_l2_loss_cells[c, :] = (evaluation_weights[c] @
                                     (basis_functions_values @ monomials_do_fs - monomials_values) ** 2)
        test_h1_loss_cells[c, :] = (evaluation_weights[c] @
                                     ((basis_functions_derivatives_values[0] @ monomials_do_fs - monomials_derivatives_values[0]) ** 2
                                     + (basis_functions_derivatives_values[1] @ monomials_do_fs - monomials_derivatives_values[1]) ** 2))

    test_l2_loss = np.sqrt(np.sum(test_l2_loss_cells, axis = 0))
    test_h1_loss = np.sqrt(np.sum(test_h1_loss_cells, axis=0))

    return test_l2_loss, test_h1_loss, test_l2_loss_cells, test_h1_loss_cells


def compute_laplacian_loss(geometry_utilities: gedim.GeometryUtilities,
                           mesh_geometric_data: MeshGeometricData2D,
                           reference_element_data: LocalSpace_PCC_2D.ReferenceElementData,
                           local_space_data: LocalSpace_PCC_2D.LocalSpaceData) -> Tuple[float, NDArray[np.float64]]:


    """
    To preserve powers of the mesh-size w.r.t. training loss,
    this loss should be computed over the rotated inertia polygon
    """

    num_cell_2 = len(mesh_geometric_data.cell2_ds_vertices)
    test_laplacian_loss_cells = np.zeros([num_cell_2])

    quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
    reference_quadrature = gedim.quadrature.Quadrature_Gauss2D_Triangle()
    reference_quadrature_data = reference_quadrature.fill_points_and_weights(20)
    evaluation_points: Dict[int, NDArray[np.float64]] = {}
    evaluation_weights: Dict[int, NDArray[np.float64]] = {}
    for c in range(num_cell_2):

        internal_quadrature = quadrature.polygon_internal_quadrature(reference_quadrature_data,
                                                                     mesh_geometric_data.cell2_ds_triangulations[
                                                                         c])

        evaluation_points[c] = internal_quadrature.points
        evaluation_weights[c] = internal_quadrature.weights

        assert geometry_utilities.is_value_zero(abs(np.sum(internal_quadrature.weights) - mesh_geometric_data.cell2_ds_areas[c]), geometry_utilities.tolerance2_d())

    evaluation_navem_input_output = None
    match reference_element_data.method_type:
        case LocalSpace_PCC_2D.MethodTypes.NAVEM:
            evaluation_navem_input_output = NAVEM_PCC_2D.create_navem_input_output(geometry_utilities,
                                                                                   mesh_geometric_data,
                                                                                   reference_element_data.navem_categories,
                                                                                   evaluation_points,
                                                                                   evaluation_weights,
                                                                                   predict_laplacian=True,
                                                                                   navem_element_type=NAVEM_PCC_2D.NAVEMMappingType.rotated_inertia)
        case _:
            raise ValueError("not valid method type")

    denominator = 0

    for c in range(num_cell_2):
        local_space_data.create_local_space(geometry_utilities.tolerance1_d(),
                                            geometry_utilities.tolerance2_d(),
                                            mesh_geometric_data,
                                            c,
                                            reference_element_data)

        basis_functions_laplacian_values \
            = local_space_data.basis_functions_laplacian_values(reference_element_data,
                                                                polydim.vem.pcc.ProjectionTypes.pi0km1_der,
                                                                evaluation_points[c],
                                                                evaluation_navem_input_output)

        test_laplacian_loss_cells[c] = 0.0
        for i in range(basis_functions_laplacian_values.shape[1]):
            test_laplacian_loss_cells[c] += np.sum(evaluation_navem_input_output[c].internal_quadrature_per_do_fs[i].weights * (basis_functions_laplacian_values[:, i] ** 2))

        denominator += basis_functions_laplacian_values.shape[1]

    test_laplacian_loss = np.sqrt(np.sum(test_laplacian_loss_cells) / denominator)

    return test_laplacian_loss, test_laplacian_loss_cells
