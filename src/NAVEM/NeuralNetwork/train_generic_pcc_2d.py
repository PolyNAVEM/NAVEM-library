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

from pypolydim import gedim
from NAVEM.Utilities.NAVEMGenerators import NAVEMGenerators
from NAVEM.PCC_2D.NAVEM_PCC_2D import NAVEMType, NAVEMElementType
from NAVEM.NeuralNetwork.training_utilities import *
from NAVEM.NeuralNetwork.h_navem_network import Flags, set_flags, HNAVEMNetworksContainer, BoundaryLoss, \
    write_flags_on_dictionary
import numpy as np
import tensorflow as tf
from NAVEM.geometry.geometry_utilities import MeshGeometricData2D
import csv
from NAVEM.Utilities.points_generator import reference_points_distribution, PointsSegmentDistributionType


def train_navem_pcc_2d_on_generic_polygon(method_order: int,
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
                                          num_points_on_each_edge: int,
                                          regularization_coefficient: float,
                                          export_training_data_file_path: str,
                                          export_training_info: bool = False,
                                          use_sqrt_in_train: bool = False,
                                          element_type: NAVEMElementType = NAVEMElementType.generic_convex,
                                          harmonic_degree: int = 20,
                                          normalization_diameter: float = 3.0,
                                          use_hanging_function: bool = True):
    assert method_order == 1
    assert method_type == NAVEMType.H_NAVEM

    network_input_dimension = 2 + 2 * (num_vertices - 1)

    navem_generators = NAVEMGenerators(geometry_utilities,
                                       num_vertices,
                                       harmonic_degree,
                                       use_hanging_function,
                                       normalization_diameter)

    num_training_polygons = mesh.cell2_d_total_number()

    flags: Flags = set_flags(network_input_dimension,
                             method_order,
                             method_type.value,
                             element_type.value,
                             num_vertices,
                             navem_generators.num_generators,
                             mesh_import_path,
                             num_training_polygons,
                             num_hidden_layers,
                             num_neurons_per_layer,
                             num_epoches_opt_order1,
                             num_epoches_opt_order2,
                             learning_rate_max,
                             learning_rate_min,
                             use_sqrt_in_train,
                             harmonic_degree,
                             normalization_diameter,
                             use_hanging_function,
                             navem_generators.list_id_vertices_hanging,
                             regularization_coefficient,
                             num_points_on_each_edge,
                             export_training_data_file_path)

    write_flags_on_dictionary(flags)

    nn = HNAVEMNetworksContainer(flags)

    # evaluation points
    reference_eval_points = reference_points_distribution(0.0, 1.0, num_points_on_each_edge,
                                                          PointsSegmentDistributionType.uniform)

    # Initialize edge loss
    boundary_loss = BoundaryLoss(geometry_utilities, method_order)
    boundary_loss.initialize_basis_evaluations_on_boundary(reference_eval_points, num_vertices)

    tangents = np.array([], dtype=np.float64).reshape(0, 2)
    labels = np.array([], dtype=np.float64)
    labels_derivatives = np.array([], dtype=np.float64)
    vertex_filter = np.array([], dtype=np.float64).reshape(0)

    super_vandermonde = np.zeros(
        (num_training_polygons * num_vertices, num_points_on_each_edge * num_vertices, navem_generators.num_generators))
    super_vander_dx = np.zeros(
        (num_training_polygons * num_vertices, num_points_on_each_edge * num_vertices, navem_generators.num_generators))
    super_vander_dy = np.zeros(
        (num_training_polygons * num_vertices, num_points_on_each_edge * num_vertices, navem_generators.num_generators))
    input_network = np.zeros((num_training_polygons * num_vertices, network_input_dimension - 2))

    exact_bfgs = True
    for c in range(num_training_polygons):

        polygon = mesh_geometric_data.cell2_ds_polygon[c]
        mapped_angles = np.expand_dims(np.array(mesh_geometric_data.cell2_ds_mapped_polygon_internal_angles[c]), axis=1)

        for v_id in range(num_vertices):

            rotated_vertices, _, _, scaling = polygon.map_f_inv(polygon.vertices, v_id)
            rotated_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)
            internal_angles = np.roll(mapped_angles, axis=1, shift=-v_id)
            vertex_distance = scaling * np.roll(polygon.mapped_max_vertex_distance, shift=-v_id)

            c_points, c_labels, c_tangents, c_labels_derivatives = boundary_loss.add_polygon(rotated_vertices,
                                                                                             add_coordinates=True)

            labels = np.concatenate([labels, c_labels])
            labels_derivatives = np.concatenate([labels_derivatives, c_labels_derivatives])
            tangents = np.concatenate([tangents, c_tangents])

            num_points = c_points.shape[0]
            vander_points = tf.transpose(c_points).numpy()
            vander_points = vander_points[0:2, :]
            zeros = np.zeros([1, num_points])
            vander_points = np.concatenate((vander_points, zeros), axis=0)

            local_vander, local_vandermonde_grads = (
                navem_generators.vander_and_vander_derivatives(vander_points,
                                                               rotated_vertices,
                                                               internal_angles=internal_angles,
                                                               vertex_distance=vertex_distance))

            super_vandermonde[c * num_vertices + v_id, :, :] = local_vander
            super_vander_dx[c * num_vertices + v_id, :, :] = local_vandermonde_grads[0, :, :]
            super_vander_dy[c * num_vertices + v_id, :, :] = local_vandermonde_grads[1, :, :]

            input_network[c * num_vertices + v_id, :] = rotated_vertices[:2, 1:].flatten()

            local_vertex_filter = np.ones((num_points,))
            for vi in range(num_vertices):
                local_vertex_filter *= np.linalg.norm(c_points[:, :2] - rotated_vertices[:2, vi], axis=1) > 4e-2
            vertex_filter = np.concatenate([vertex_filter, local_vertex_filter])

    labels = tf.squeeze(tf.convert_to_tensor(labels, dtype=tf.float64))
    labels_derivatives = tf.convert_to_tensor(labels_derivatives, dtype=tf.float64)
    tangents = tf.convert_to_tensor(tangents, dtype=tf.float64)
    vertex_filter = tf.convert_to_tensor(vertex_filter, dtype=tf.float64)

    learning_rate_scheduler = get_lr_scheduler(num_epoches_opt_order1, learning_rate_min, learning_rate_max)
    exponential_learning_rate = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)
    cb_list = [exponential_learning_rate]

    list_of_losses = []
    list_of_times = []
    if export_training_info:
        store_loss = StoreLoss(list_of_losses, n_steps=1)
        store_time = StoreTime(list_of_times, n_steps=1)
        cb_list = [exponential_learning_rate, store_loss, store_time]

    nn.nn_basis_function.train_vander.assign(tf.convert_to_tensor(super_vandermonde, dtype=tf.float64))

    training_input = tf.convert_to_tensor(input_network, dtype=tf.float64)
    labels = tf.reshape(labels, (num_training_polygons * num_vertices, num_vertices * num_points_on_each_edge))

    considered_loss = nn.nn_basis_function.copy_value
    tf.print("\n---------- Start Training ----------\n")

    tf.print("------------------- Copy of the basis functions -------------------\n")
    pol_results = train_adam_bfgs(nn.nn_basis_function, considered_loss, training_input, labels, num_epoches_opt_order1,
                                  num_epoches_opt_order2, cb_list, use_bfgs=exact_bfgs)

    tf.print("\n---------- Copy of the basis functions derivatives ----------\n")
    nn.nn_basis_derivatives.copy_weights(nn.nn_basis_function)

    nn.nn_basis_derivatives.train_vander_dx.assign(tf.convert_to_tensor(np.array(super_vander_dx), dtype=tf.float64))
    nn.nn_basis_derivatives.train_vander_dy.assign(tf.convert_to_tensor(np.array(super_vander_dy), dtype=tf.float64))
    nn.nn_basis_derivatives.deriv_labels.assign(
        tf.reshape(labels_derivatives, (num_training_polygons * num_vertices, num_vertices * num_points_on_each_edge)))
    nn.nn_basis_derivatives.tangent_x.assign(
        tf.reshape(tangents[:, 0], (num_training_polygons * num_vertices, num_vertices * num_points_on_each_edge)))
    nn.nn_basis_derivatives.tangent_y.assign(
        tf.reshape(tangents[:, 1], (num_training_polygons * num_vertices, num_vertices * num_points_on_each_edge)))
    nn.nn_basis_derivatives.vertex_filter.assign(
        tf.reshape(vertex_filter, (num_training_polygons * num_vertices, num_vertices * num_points_on_each_edge)))
    considered_loss_dx = nn.nn_basis_derivatives.copy_grad

    learning_rate_scheduler = get_lr_scheduler(num_epoches_opt_order1, learning_rate_min, learning_rate_max)
    exponential_learning_rate = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)
    cb_list = [exponential_learning_rate]

    list_of_losses_deriv = []
    list_of_times_deriv = []
    if export_training_info:
        store_loss = StoreLoss(list_of_losses_deriv, n_steps=1)
        store_time = StoreTime(list_of_times_deriv, n_steps=1)
        cb_list = [exponential_learning_rate, store_loss, store_time]

    pol_deriv_results = train_adam_bfgs(nn.nn_basis_derivatives, considered_loss_dx, training_input, labels,
                                        num_epoches_opt_order1, num_epoches_opt_order2, cb_list,
                                        use_bfgs=exact_bfgs)

    if export_training_info:
        adam_steps = np.arange(0, len(list_of_losses))
        bfgs_steps = np.arange(len(list_of_losses), len(list_of_losses) + len(pol_results[1].losses))

        steps = np.concatenate([adam_steps, bfgs_steps], axis=0)
        losses = np.concatenate([np.array(list_of_losses), np.array(pol_results[1].losses)], axis=0)
        times = np.concatenate([np.array(list_of_times), list_of_times[-1] + np.array(pol_results[1].times)], axis=0)

        fake_header = np.array([[num_epoches_opt_order1, len(pol_results[1].losses), 999999]])
        export_data = np.stack([steps, losses, times], axis=1)

        adam_steps_deriv = np.arange(0, len(list_of_losses_deriv))
        bfgs_steps_deriv = np.arange(len(list_of_losses_deriv),
                                     len(list_of_losses_deriv) + len(pol_deriv_results[1].losses))

        steps_deriv = np.concatenate([adam_steps_deriv, bfgs_steps_deriv], axis=0)
        losses_deriv = np.concatenate([np.array(list_of_losses_deriv), np.array(pol_deriv_results[1].losses)], axis=0)
        times_deriv = np.concatenate(
            [np.array(list_of_times_deriv), list_of_times_deriv[-1] + np.array(pol_deriv_results[1].times)], axis=0)
        export_data_deriv = np.stack([steps_deriv, losses_deriv, times_deriv], axis=1)
        data = np.concatenate([fake_header, export_data, export_data_deriv], axis=0)

        np.savetxt(export_training_data_file_path + '/train_info.csv', data, delimiter=';')

    # Storing the final loss and the l2 and h1 errors
    with open('{}/loss.csv'.format(flags['name_storage']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['NPolygons', 'L2 loss', 'H1 loss'])

        basis_coefficients = nn.nn_basis_function.call(training_input)
        considered_loss(labels, basis_coefficients)

        derivatives_coefficients = nn.nn_basis_derivatives.call(training_input)
        considered_loss_dx(labels, derivatives_coefficients)

        writer.writerow([num_training_polygons,
                         np.sqrt(nn.nn_basis_function.curr_distance_l2.numpy()),
                         np.sqrt(nn.nn_basis_derivatives.curr_distance_h1.numpy())])

    # Final storage
    nn.save_model()
