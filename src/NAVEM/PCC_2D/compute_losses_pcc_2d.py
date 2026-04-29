import numpy as np
import tensorflow as tf
from src.NAVEM.PCC_2D import NAVEM_PCC_2D, LocalSpace_PCC_2D
from pypolydim import gedim, polydim
from src.GeDiM.geometry.geometry_utilities import MeshGeometricData2D
from typing import Dict, Tuple
from numpy.typing import NDArray


def compute_polynomial_loss(geometry_utilities: gedim.GeometryUtilities,
                            mesh_geometric_data: MeshGeometricData2D,
                            reference_element_data: LocalSpace_PCC_2D.ReferenceElementData,
                            method_type: LocalSpace_PCC_2D.MethodTypes) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

    num_cell_2 = len(mesh_geometric_data.mesh_geometric_data.cell2_ds_vertices)

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
                                                                     mesh_geometric_data.mesh_geometric_data.cell2_ds_triangulations[
                                                                         c])

        evaluation_points[c] = internal_quadrature.points
        evaluation_weights[c] = internal_quadrature.weights

        assert geometry_utilities.is_value_zero(abs(np.sum(internal_quadrature.weights) - mesh_geometric_data.mesh_geometric_data.cell2_ds_areas[c]), geometry_utilities.tolerance2_d())

    local_space_data = LocalSpace_PCC_2D.LocalSpaceData(geometry_utilities, mesh_geometric_data, reference_element_data)
    evaluation_navem_input_output = None
    match method_type:
        case LocalSpace_PCC_2D.MethodTypes.NAVEM | LocalSpace_PCC_2D.MethodTypes.B_NAVEM | LocalSpace_PCC_2D.MethodTypes.P_NAVEM:
            evaluation_navem_input_output = NAVEM_PCC_2D.create_navem_input_output(geometry_utilities,
                                                                                   mesh_geometric_data,
                                                                                   NAVEM_PCC_2D.NAVEMType(
                                                                                       method_type.value),
                                                                                   reference_element_data.navem_categories,
                                                                                   evaluation_points)
        case _:
            pass

    for c in range(num_cell_2):
        local_space_data.create_local_space(geometry_utilities.tolerance1_d(),
                                            geometry_utilities.tolerance2_d(),
                                            mesh_geometric_data,
                                            c,
                                            reference_element_data)

        polygon_vertices = mesh_geometric_data.mesh_geometric_data.cell2_ds_vertices[c]
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

    print(test_l2_loss, test_h1_loss)

    return test_l2_loss, test_h1_loss, test_l2_loss_cells, test_h1_loss_cells