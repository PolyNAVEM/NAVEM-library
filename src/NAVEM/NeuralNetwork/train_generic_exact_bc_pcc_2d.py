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

from NAVEM.PCC_2D.NAVEM_PCC_2D import NAVEMType
from NAVEM.geometry.mesh_utilities import MeshGeometricData2D
from NAVEM.Utilities.NAVEMPolygon import NAVEMPolygon
from NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import *
from NAVEM.NeuralNetwork.b_navem_network import BNAVEMNetwork
from NAVEM.NeuralNetwork.p_navem_network import PNAVEMNetwork
from NAVEM.NeuralNetwork.training_utilities import *
from NAVEM.PCC_2D.NAVEM_Data_PCC_2D import NAVEMElementType, BasisFunctionType
import csv
from NAVEM.Utilities.points_generator import *
from NAVEM.Utilities.enforcing_boundary_functions import BubbleType, BoundaryMethodType
from NAVEM.Utilities.PrintInformation import print_training_information


def train_exact_bc_navem_pcc_2d_on_generic_polygon(method_order: int,
                                                   basis_function_type: BasisFunctionType,
                                                   method_type: NAVEMType,
                                                   num_vertices: int,
                                                   geometry_utilities: gedim.GeometryUtilities,
                                                   mesh: gedim.MeshMatricesDAO,
                                                   mesh_geometric_data: MeshGeometricData2D,
                                                   mesh_import_path: str,
                                                   num_hidden_layers: int,
                                                   num_neurons_per_layer: int,
                                                   num_epoches_opt_order1: int,
                                                   num_epoches_opt_order2: int,
                                                   learning_rate_max: float,
                                                   learning_rate_min: float,
                                                   quadrature_order: int,
                                                   distribution_points_type: int,
                                                   regularization_coefficient: float,
                                                   export_training_data_file_path: str,
                                                   export_training_info: bool = False,
                                                   copy_basis_in_train: bool = False,
                                                   use_sqrt_in_train: bool = False,
                                                   element_type: NAVEMElementType = NAVEMElementType.generic_convex,
                                                   boundary_method_type_g: BoundaryMethodType = BoundaryMethodType.segment,
                                                   boundary_method_type_adf: BoundaryMethodType = BoundaryMethodType.segment,
                                                   bubble_type: BubbleType = BubbleType.approximate_distance_function):
    assert method_order == 1

    network_input_dimension = 2 + 2 * (num_vertices - 1)
    num_training_polygons = mesh.cell2_d_total_number()
    num_functions_per_polygon = num_vertices - (method_type == NAVEMType.P_NAVEM)

    flags: Flags = set_flags(network_input_dimension,
                             method_order,
                             method_type.value,
                             basis_function_type.value,
                             num_functions_per_polygon,
                             element_type.value,
                             num_vertices,
                             mesh_import_path,
                             num_training_polygons,
                             num_hidden_layers,
                             num_neurons_per_layer,
                             num_epoches_opt_order1,
                             num_epoches_opt_order2,
                             learning_rate_max,
                             learning_rate_min,
                             use_sqrt_in_train,
                             regularization_coefficient,
                             export_training_data_file_path,
                             copy_basis_in_train,
                             quadrature_order,
                             distribution_points_type,
                             boundary_method_type_g.value,
                             boundary_method_type_adf.value,
                             bubble_type.value)

    write_flags_on_dictionary(flags)
    print_training_information(flags)

    nn = None
    match method_type:
        case NAVEMType.P_NAVEM:
            nn = PNAVEMNetwork(flags, in_training=True)
        case NAVEMType.B_NAVEM:
            nn = BNAVEMNetwork(flags, in_training=True)
        case _:
            raise ValueError("not valid method")

    distribution_points_type = PointsTriangleDistributionType(distribution_points_type)
    uniform_boundary = False
    uniform_scale = True
    accumulating_power = 0.75
    accumulating_border = BorderType.no_borders
    reference_nodes = grid_over_triangle(PointsTriangleDistributionType(distribution_points_type), quadrature_order,
                                         uniform_boundary=uniform_boundary, uniform_rescale=uniform_scale,
                                         accumulating_power=accumulating_power, accumulating_border=accumulating_border,
                                         polygon=num_vertices > 3)

    num_points_per_polygon = reference_nodes.shape[1] * num_vertices

    inputs = np.zeros((num_training_polygons * num_points_per_polygon * num_functions_per_polygon,
                       network_input_dimension))

    xy_per_pol = np.zeros(shape=(num_training_polygons, num_points_per_polygon, 2))
    vertices_per_pol = np.zeros(shape=(num_training_polygons, 2, num_vertices))
    jac_per_pol = np.zeros(shape=(num_training_polygons, 2, 2, num_functions_per_polygon))
    jac_inv_per_pol = np.zeros(shape=(num_training_polygons, 2, 2, num_functions_per_polygon))

    exact_bfgs = True

    for c in range(num_training_polygons):

        internal_point = mesh_geometric_data.mesh_geometric_data.cell2_ds_centroids[c]
        polygon = mesh_geometric_data.cell2_ds_polygon[c]

        inertia_mapped_internal_point, _ = polygon.map_inertia_inv(np.expand_dims(internal_point, axis=1))
        inertia_mapped_list_triangles = geometry_utilities.polygon_triangulation_by_internal_point(
            polygon.mapped_vertices, inertia_mapped_internal_point)
        inertia_mapped_list_triangles_points = geometry_utilities.extract_triangulation_points_by_internal_point(
            polygon.mapped_vertices, inertia_mapped_internal_point, inertia_mapped_list_triangles)

        # Loss will be minimized over the inertia polygon
        inertia_mapped_internal_nodes = grid_over_polygon(distribution_points_type,
                                                          reference_nodes,
                                                          inertia_mapped_list_triangles_points,
                                                          uniform_boundary, accumulating_border)

        xy_per_pol[c, :, :] = inertia_mapped_internal_nodes[:2, :].T  # points over the inertia polygon
        vertices_per_pol[c, :, :] = polygon.mapped_vertices[:2, :]  # vertices are set up over the inertia polygon

        for v_id in range(num_functions_per_polygon):
            rotated_vertices, rotated_mapped_points, jac_inv, jac, _ = NAVEMPolygon.map_fix_vertex_inv(
                polygon.mapped_vertices,
                inertia_mapped_internal_nodes,
                v_id)
            rotated_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)

            flat_rotated_vertices = rotated_vertices[:2, 1:].flatten()
            rep_flat_rot_vertices = np.tile(flat_rotated_vertices[np.newaxis, :], [num_points_per_polygon, 1])
            c_inputs = np.concatenate([rotated_mapped_points[:2, :].T, rep_flat_rot_vertices], axis=1)

            inputs[(c * num_functions_per_polygon + v_id) * num_points_per_polygon:
                   (c * num_functions_per_polygon + v_id + 1) * num_points_per_polygon, :] = c_inputs
            jac_per_pol[c, :, :, v_id] = jac[:2, :2]
            jac_inv_per_pol[c, :, :, v_id] = jac_inv[:2, :2]

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
    labels = 0 * inputs[:, 0:1]

    setup_n_derivatives = SetupDerivatives.basis
    considered_loss = None
    match method_type:
        case NAVEMType.B_NAVEM:
            setup_n_derivatives = SetupDerivatives.basis_and_derivatives_and_laplacian
            considered_loss = nn.pinn_laplace_loss
        case NAVEMType.P_NAVEM:
            nn.store_terms_for_loss(xy_per_pol, vertices_per_pol, jac_inv_per_pol)
            setup_n_derivatives = SetupDerivatives.basis_and_derivatives
            if copy_basis_in_train:
                considered_loss = nn.internal_pol_and_grad_loss
            else:
                considered_loss = nn.internal_grad_loss
        case _:
            raise ValueError("not valid navem type")

    nn.setup_model_global_input(xy_per_pol, vertices_per_pol, jac_per_pol, setup_n_derivatives, geometry_utilities)

    learning_rate_scheduler = get_lr_scheduler(num_epoches_opt_order1, learning_rate_min, learning_rate_max)
    exponential_learning_rate = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)
    cb_list = [exponential_learning_rate]

    list_of_losses = []
    list_of_times = []
    if export_training_info:
        store_loss = StoreLoss(list_of_losses, n_steps=1)
        store_time = StoreTime(list_of_times, n_steps=1)
        cb_list = [exponential_learning_rate, store_loss, store_time]

    tf.print("\n---------- Start training ----------\n")

    results = None
    match method_type:
        case NAVEMType.B_NAVEM:
            del vertices_per_pol
            del jac_per_pol
            del jac_inv_per_pol
            del xy_per_pol
            del mesh_geometric_data
            del mesh

            results = train_adam_bfgs(nn, considered_loss, inputs, labels, num_epoches_opt_order1,
                                      num_epoches_opt_order2, cb_list, use_bfgs=exact_bfgs)
        case NAVEMType.P_NAVEM:
            results = train_adam_bfgs(nn, considered_loss, inputs, labels, num_epoches_opt_order1,
                                      num_epoches_opt_order2, cb_list, use_bfgs=exact_bfgs)
        case _:
            raise ValueError("not valid navem type")

    if export_training_info:
        adam_steps = np.arange(0, len(list_of_losses))
        bfgs_steps = np.arange(len(list_of_losses), len(list_of_losses) + len(results[1].losses))

        steps = np.concatenate([adam_steps, bfgs_steps], axis=0)
        losses = np.concatenate([np.array(list_of_losses), np.array(results[1].losses)], axis=0)
        times = np.concatenate([np.array(list_of_times), list_of_times[-1] + np.array(results[1].times)], axis=0)

        fake_header = np.array([[num_epoches_opt_order1, len(results[1].losses), 999999]])
        export_data = np.stack([steps, losses, times], axis=1)

        data = np.concatenate([fake_header, export_data], axis=0)

        np.savetxt(export_training_data_file_path + '/train_info.csv', data, delimiter=';')

    # Storing the final loss and the l2 and h1 errors
    with open('{}/loss.csv'.format(flags['name_storage']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['NPolygons', 'Laplacian loss'])

        nn_output = nn.call(inputs)
        considered_loss(labels, nn_output)

        match method_type:
            case NAVEMType.B_NAVEM:
                writer.writerow([num_training_polygons, np.sqrt(nn.curr_laplacian.numpy())])
            case NAVEMType.P_NAVEM:
                writer.writerow([num_training_polygons,
                                 nn.loss_one.numpy(),
                                 nn.loss_x.numpy(),
                                 nn.loss_y.numpy()])
            case _:
                raise ValueError("not valid navem type")

    # Final storage
    nn.save_model()
