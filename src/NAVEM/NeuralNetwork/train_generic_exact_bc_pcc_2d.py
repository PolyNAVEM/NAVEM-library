from pypolydim import gedim
from src.NAVEM.Utilities.NAVEM_PCC_2D import NAVEMType
from src.GeDiM.geometry.geometry_utilities import MeshGeometricData2D
import numpy as np
from src.NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import *
from src.NAVEM.NeuralNetwork.b_navem_network import BNAVEMNetwork
from src.NAVEM.NeuralNetwork.p_navem_super_network import PNAVEMSupernetwork
from src.NAVEM.NeuralNetwork.training_utilities import *
import csv
from src.NAVEM.Utilities.points_generator import *


def write_dictionary(flags: Flags) -> None:

    file = open('{}/dictionary.txt'.format(flags['name_storage']), 'w')
    file.write("network_input_dimension = {}\n".format(flags['input_dim']))
    file.write("output_dim = {}\n".format(flags['output_dim']))
    file.write("method_order = {}\n".format(flags['method_order']))
    file.write("method_type = '{}'\n".format(flags['method_type']))
    file.write("num_vertices = {}\n".format(flags['num_vertices']))
    file.write("num_training_polygons = {}\n".format(flags['num_training_polygons']))
    file.write("regularization_coefficient = {}\n".format(flags['regularization_coefficient']))
    file.write("mesh_import_path = '{}'\n".format(flags['mesh_import_path']))
    file.write("quadrature_order = '{}'\n".format(flags['quadrature_order']))
    file.write("distribution_points_type = {}\n".format(flags['distribution_points_type']))

    file.write("num_hidden_layers = {}\n".format(flags['num_hidden_layers']))
    file.write("num_neurons_per_layer = {}\n".format(flags['num_neurons_per_layer']))
    file.write("num_epoches_opt_order1 = {}\n".format(flags['num_epoches_opt_order1']))
    file.write("num_epoches_opt_order2 = {}\n".format(flags['num_epoches_opt_order2']))
    file.write("learning_rate_max = {}\n".format(flags['learning_rate_max']))
    file.write("learning_rate_min = {}\n".format(flags['learning_rate_min']))

    if flags['p_navem_exact_constant']:
        file.write("p_navem_exact_constant = True\n")
    else:
        file.write("p_navem_exact_constant = False\n")

    file.close()

def train_exact_bc_navem_pcc_2d_on_generic_polygon(method_order: int,
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
                                                   p_navem_exact_constant: bool,
                                                   regularization_coefficient: float,
                                                   export_training_data_file_path: str,
                                                   export_training_info: bool = False,
                                                   copy_basis_in_train: bool = False,
                                                   use_sqrt_in_train: bool = False):

    assert method_order == 1

    network_input_dimension = 2 + 2 * (num_vertices - 1)
    num_training_polygons = mesh.cell2_d_total_number()

    flags: Flags = set_flags(network_input_dimension,
                             method_order,
                             method_type.value,
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
                             p_navem_exact_constant,
                             quadrature_order,
                             distribution_points_type)

    write_dictionary(flags)

    match method_type:
        case NAVEMType.P_NAVEM:
            nn = PNAVEMSupernetwork(flags)
        case NAVEMType.B_NAVEM:
            nn = BNAVEMNetwork(flags)
        case _:
            raise ValueError("not valid method")

    distribution_points_type = PointsTriangleDistributionType(distribution_points_type)
    reference_nodes = grid_over_triangle(PointsTriangleDistributionType(distribution_points_type), quadrature_order,
                                         uniform_boundary = False, uniform_rescale = True,
                                         accumulating_power = 1.0, accumulating_border = BorderType.no_borders,
                                         polygon=num_vertices > 3)

    num_points_per_polygon = reference_nodes.shape[1] * num_vertices

    num_functions_per_polygon = num_vertices - p_navem_exact_constant

    inputs = np.zeros((num_training_polygons * num_points_per_polygon * num_functions_per_polygon, network_input_dimension))
    training_quad_w = np.zeros((num_training_polygons * num_points_per_polygon * num_functions_per_polygon, 1))

    xy_per_pol = np.zeros(shape=(num_training_polygons, num_points_per_polygon, 2))
    vertices_per_pol = np.zeros(shape=(num_training_polygons, 2, num_vertices))
    jac_per_pol = np.zeros(shape=(num_training_polygons, 2, 2, num_functions_per_polygon))
    jac_inv_per_pol = np.zeros(shape=(num_training_polygons, 2, 2, num_functions_per_polygon))

    exact_bfgs = True

    for c in range(num_training_polygons):

        vertices_idxs = train_mesh.cells_2[c].vertices
        hanging_vertices = geometric_data[c].id_aligned_vertices
        internal_point = train_mesh.cells_2[c].internal_point
        cell_vertices = geometric_data[c].vertex_points
        diameter = geometric_data[c].diameter
        centroid = geometric_data[c].centroid
        polygon = geometric_data[c].polygon



        inertia_mapped_internal_point, _ = polygon.map_inertia_inv(internal_point)
        inertia_mapped_list_triangles = geometry_utilities.compute_triangulation_by_internal_point(
            inertia_mapped_internal_point, polygon.mapped_vertices)

        if pts_type == "gauss_triangle":
            inertia_mapped_internal_nodes, inertia_mapped_internal_weights = internal_quadrature(quadrature,
                                                                                                 inertia_mapped_list_triangles)
        elif pts_type == "uniform":
            inertia_mapped_internal_nodes = nodes_generator.uniform_grid_over_polygon(quadrature_order,
                                                                                      inertia_mapped_list_triangles,
                                                                                      boundary=border_pts, rescale=True)
        else:
            inertia_mapped_internal_nodes = nodes_generator.concentrating_grid_over_polygon(quadrature_order,
                                                                                            inertia_mapped_list_triangles,
                                                                                            power=0.75, border_type=1)

        xy_per_pol[c, :, :] = inertia_mapped_internal_nodes[:2, :].T
        vertices_per_pol[c, :, :] = polygon.mapped_vertices[:2, :]

        for v_id in range(num_functions_per_polygon):
            rotated_vertices, resc_mapped_points, jac_inv, jac, inv_rescaling_factor = map_fix_vertex_inv(
                polygon.mapped_vertices,
                inertia_mapped_internal_nodes,
                v_id)
            rotated_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)

            flat_rotated_vertices = rotated_vertices[:2, 1:].flatten()
            rep_flat_rot_vertices = np.tile(flat_rotated_vertices[np.newaxis, :], [num_points_per_polygon, 1])
            c_inputs = np.concatenate([resc_mapped_points[:2, :].T, rep_flat_rot_vertices], axis=1)

            inputs[(c * num_functions_per_polygon + v_id) * num_points_per_polygon: (c * num_functions_per_polygon + v_id + 1) * num_points_per_polygon, :] = c_inputs
            jac_per_pol[c, :, :, v_id] = jac[:2, :2]
            jac_inv_per_pol[c, :, :, v_id] = jac_inv[:2, :2]

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
    labels = 0 * inputs[:, 0:1]

    if method == "BNAVEM":
        setup_n_derivatives = 2
        considered_loss = nn.pinn_laplace_loss
    elif method == "NAPEM":
        nn.nn_func.store_terms_for_loss(xy_per_pol, vertices_per_pol, jac_inv_per_pol, napem_exact_one)
        setup_n_derivatives = 1
        if napem_exact_one:
            nn.nn_func.copy_pols = nn.nn_func.copy_pols_exact_one
            nn.nn_func.copy_grads = nn.nn_func.copy_grads_exact_one

        if use_grad_in_train and copy_pols_in_train:
            nn.nn_func.call_for_training = nn.nn_func.get_u_and_du
            considered_loss = nn.nn_func.internal_pol_and_grad_loss
        elif use_grad_in_train and not copy_pols_in_train:
            nn.nn_func.call_for_training = nn.nn_func.get_du
            considered_loss = nn.nn_func.internal_grad_loss
        elif not use_grad_in_train and copy_pols_in_train:
            nn.nn_func.call_for_training = nn.nn_func.get_u
            considered_loss = nn.nn_func.internal_pol_loss
        else:
            raise ValueError("At least one between use_grad_in_train and copy_pols_in_train must be True")


    nn.setupModel_global_input(xy_per_pol, vertices_per_pol, jac_per_pol, setup_n_derivatives, geometry_utilities)

    expCyclLr = ExpCyclicLr(num_epoches_opt_order1, learning_rate_min, learning_rate_max)
    cb_list = [expCyclLr]

    if export_training_info:
        list_of_losses = []
        list_of_times = []
        storeLoss = StoreLoss(list_of_losses, n_steps=1)
        storeTime = StoreTime(list_of_times, n_steps=1)
        cb_list = [expCyclLr, storeLoss, storeTime]

    tf.print("\n---------- Start training ----------\n")

    if method == "BNAVEM":
        del vertices_per_pol
        del jac_per_pol
        del jac_inv_per_pol
        del xy_per_pol
        del geometric_data
        del train_mesh

        results = train_adam_bfgs(nn, considered_loss, inputs, labels, num_epoches_opt_order1,
                                           num_epoches_opt_order2, cb_list, use_bfgs=exact_bfgs)
    else:
        results = train_adam_bfgs(nn.nn_func, considered_loss, inputs, labels, num_epoches_opt_order1,
                                           num_epoches_opt_order2, cb_list, use_bfgs=exact_bfgs)

        tf.print("\n---------- Start gradient training ----------\n")
        func_pred_grad = nn.nn_func.get_u_and_du(inputs)[:, 1:]
        nn.nn_grad.store_terms_for_loss(func_pred_grad, vertices_per_pol, jac_inv_per_pol)
        expCyclLr = ExpCyclicLr(num_epoches_opt_order1, learning_rate_min, learning_rate_max)
        cb_list = [expCyclLr]

        if export_training_info:
            list_of_losses_deriv = []
            list_of_times_deriv = []
            storeLoss = StoreLoss(list_of_losses_deriv, n_steps=1)
            storeTime = StoreTime(list_of_times_deriv, n_steps=1)
            cb_list = [expCyclLr, storeLoss, storeTime]

        results2 = train_adam_bfgs(nn.nn_grad, nn.nn_grad.inter_grad_loss, inputs, labels, num_epoches_opt_order1,
                                   num_epoches_opt_order2, cb_list, use_bfgs=exact_bfgs)

    if export_training_info:
        adam_steps = np.arange(0, len(list_of_losses))
        bfgs_steps = np.arange(len(list_of_losses), len(list_of_losses)+len(results[1].losses))

        steps = np.concatenate([adam_steps, bfgs_steps], axis=0)
        losses = np.concatenate([np.array(list_of_losses), np.array(results[1].losses)], axis=0)
        times = np.concatenate([np.array(list_of_times), list_of_times[-1] + np.array(results[1].times)], axis=0)

        fake_header = np.array([[num_epoches_opt_order1, len(results[1].losses), 999999]])
        export_data = np.stack([steps, losses, times], axis=1)

        if split_deriv and method == "NAPEM":
            adam_steps_deriv = np.arange(0, len(list_of_losses_deriv))
            bfgs_steps_deriv = np.arange(len(list_of_losses_deriv), len(list_of_losses_deriv)+len(results2[1].losses))

            steps_deriv = np.concatenate([adam_steps_deriv, bfgs_steps_deriv], axis=0)
            losses_deriv = np.concatenate([np.array(list_of_losses_deriv), np.array(results2[1].losses)], axis=0)
            times_deriv = np.concatenate([np.array(list_of_times_deriv), list_of_times_deriv[-1] + np.array(results2[1].times)], axis=0)
            export_data_deriv = np.stack([steps_deriv, losses_deriv, times_deriv], axis=1)
            data = np.concatenate([fake_header, export_data, export_data_deriv], axis=0)
        else:
            data = np.concatenate([fake_header, export_data], axis=0)

        np.savetxt('NeuralNetwork/savedModels/'+method+'_train_info.csv', data, delimiter='; ')


    # Storing the final loss and the l2 and h1 errors
    with open('{}_loss.csv'.format(flags['name_storage']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['NPolygons', 'Laplacian loss'])

        nn_output = nn.call(inputs)
        considered_loss(labels, nn_output)
        if method == "BNAVEM":
            writer.writerow([num_training_polygons,
                             np.sqrt(nn.curr_laplacian.numpy())])
        elif method == "NAPEM":
            writer.writerow([num_training_polygons,
                             nn.nn_func.loss_one.numpy(),
                             nn.nn_func.loss_x.numpy(),
                             nn.nn_func.loss_y.numpy()])
    # Final storage
    nn.saveModel()
