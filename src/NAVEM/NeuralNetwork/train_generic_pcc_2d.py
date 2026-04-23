from pypolydim import gedim
from src.NAVEM.Utilities.NAVEMPolygon import NAVEMPolygon
from src.NAVEM.Utilities.NAVEM_PCC_2D import NAVEMType
from typing import List
from src.NAVEM.NeuralNetwork.navem_network import Flags, set_flags, NAVEMNetwork
import numpy as np
import tensorflow as tf

def write_dictionary(network_input_dimension: int,
                     method_order: int,
                     num_vertices: int,
                     num_hidden_layers: int,
                     num_neurons_per_layer: int,
                     harmonic_degree: int,
                     use_hanging_function: bool,
                     num_rational_poles_per_vertex: int,
                     list_id_vertices_rationals: List[int],
                     regularization_coefficient: float,
                     export_training_data_file_path: str):

    file = open('{}_dictionary.txt'.format(FLAGS['name_storage']), 'w')
    file.write("network_input_dimension = {}\n".format(FLAGS['input_dim']))
    file.write("output_dim = {}\n".format(FLAGS['output_dim']))
    file.write("vem_k = {}\n".format(FLAGS['vem_order']))
    file.write("n_vertices = {}\n".format(FLAGS['n_edges']))
    file.write("n_edges = {}\n".format(n_vertices))
    file.write("n_training_polygons = {}\n".format(n_training_polygons))
    file.write("edge_normalization_square = {}\n".format(edge_normalization_square))
    file.write("net_reg_coef = {}\n".format(net_reg_coef))

    if read_mesh:
        file.write("read_mesh = True\n")
    else:
        file.write("read_mesh = False\n")

    file.write("name_mesh = '{}'\n".format(name_mesh))
    file.write("n_points_on_each_edge = {}\n".format(n_points_on_each_edge))
    file.write("name = '{}'\n".format(name))
    file.write("n_pol_hidden_layers = {}\n".format(FLAGS['n_pol_hidden_layers']))
    file.write("n_pol_neurons = {}\n".format(FLAGS['n_pol_neurons']))
    file.write("n_pol_epochs_order1 = {}\n".format(n_pol_epochs_order1))
    file.write("n_pol_epochs_order2 = {}\n".format(n_pol_epochs_order2))
    file.write("lr_max = {}\n".format(lr_min))
    file.write("lr_min = {}\n".format(lr_min))
    file.write("concave_convex_both = {}\n".format(concave_convex_both))

    file.write("tol_almost_hanging_nodes = {}\n".format(tol_almost_hanging_nodes))
    file.write("use_almost_hanging = {}\n".format(use_almost_hanging))

    file.write("harm_deg = {}\n".format(harm_deg))
    file.write("harmonic_type_function = '{}'\n".format(harmonic_type_function))

    if use_hanging_function:
        file.write("use_hanging_function = True\n")
        file.write("hanging_num_rat_points = {}\n".format(navem_polynomial.hanging_function.flat_poles.shape[1]))
        file.write("hanging_harm_deg = {}\n".format(navem_polynomial.hanging_function.harm_deg))
        file.write("hanging_type_rational = '{}'\n".format(navem_polynomial.hanging_function.function_type))
        file.write("hanging_type_harmonic = '{}'\n".format(navem_polynomial.hanging_function.function_type))
        file.write("list_id_vertices_hanging = [{}".format(list_id_vertices_hanging[0]))
        for id in range(len(list_id_vertices_hanging) - 1):
            file.write(", {}".format(list_id_vertices_hanging[id + 1]))
        file.write("]\n")
    else:
        file.write("use_hanging_function = False\n")
        file.write("hanging_harm_deg = {}\n".format(navem_polynomial.hanging_function.harm_deg))
        file.write("hanging_num_rat_points = {}\n".format(navem_polynomial.hanging_function.flat_poles.shape[1]))
        file.write("hanging_type_rational = '{}'\n".format(navem_polynomial.hanging_function.function_type))
        file.write("hanging_type_harmonic = '{}'\n".format(navem_polynomial.hanging_function.function_type))
        file.write("list_id_vertices_hanging = []\n")

    if use_rational_function:
        file.write("use_rational_function = True\n")
        file.write("rational_num_rat_points = {}\n".format(rational_num_rat_points))
        file.write("rational_type_function = '{}'\n".format(rational_type_function))
        file.write("list_id_vertices_rationals = [{}".format(list_id_vertices_rationals[0]))
        for id in range(len(list_id_vertices_rationals) - 1):
            file.write(", {}".format(list_id_vertices_rationals[id + 1]))
        file.write("]\n")
    else:
        file.write("use_rational_function = False\n")
        file.write("rational_num_rat_points = {}\n".format(rational_num_rat_points))
        file.write("rational_type_function = '{}'\n".format(rational_type_function))
        file.write("list_id_vertices_rationals = []\n")

    file.write("split_deriv = {}\n".format(split_deriv))
    if use_grad_in_train:
        file.write("use_grad_in_train = True\n")
    else:
        file.write("use_grad_in_train = False\n")

    if use_sqrt_in_train:
        file.write("use_sqrt_in_train = True\n")
    else:
        file.write("use_sqrt_in_train = False\n")

    file.close()


def train_exact_bc_navem_pcc_2d_on_generic_polygon(method_order: int, method_type: NAVEMType,
                                                   num_vertices: int,
                                                   geometry_utilities: gedim.GeometryUtilities,
                                                   mesh: gedim.MeshMatricesDAO,
                                                   mesh_geometric_data: gedim.MeshUtilities.MeshGeometricData2D,
                                                   num_hidden_layers: int,
                                                   num_neurons_per_layer: int,
                                                   num_epoches_opt_order1: int,
                                                   num_epoches_opt_order2: int,
                                                   learning_rate_opt_order1: float, learning_rate_opt_order2: float,
                                                   num_points_on_each_edge: int,
                                                   regularization_coefficient: float,
                                                   export_training_data_file_path: str,
                                                   harmonic_degree: int = 20,
                                                   use_hanging_function: bool = True):

    assert method_order == 1
    assert method_type == NAVEMType.NAVEM

    nkm2_polynomial = int(0.5 * vem_k * (vem_k - 1))
    n_dofs = vem_k * n_vertices + nkm2_polynomial  # boundary + internal dofs
    index_bin_size = int(np.floor(np.log2(n_dofs - 1)) + 1)
    network_input_dimension = 2 + 2 * (n_vertices - 1)

    tf.print('\n')
    tf.print('The total number of dof per element is: {}'.format(n_dofs))
    tf.print('\n')

    # set geometry tolerance
    tol = 1.0e-12
    geometry_utilities = GeometryUtilities(tol, tol_almost_hanging_nodes)
    if concave_convex_both == 0 or concave_convex_both == 2:
        if use_hanging_function:
            list_id_vertices_hanging = list(np.arange(0, n_vertices))
        else:
            list_id_vertices_hanging = []

        if use_rational_function:
            list_id_vertices_rationals = list(np.arange(0, n_vertices))
        else:
            list_id_vertices_rationals = []
    else:
        if use_hanging_function:
            list_id_vertices_hanging = [0, 1, n_vertices - 1]
        else:
            list_id_vertices_hanging = []

        if use_rational_function:
            list_id_vertices_rationals = [0, 1, n_vertices - 1]
        else:
            list_id_vertices_rationals = []

    hang_func = LaplaceSolver('polygon_hanging')
    navem_polynomial = NAVEMPolynomialBasis(geometry_utilities, vem_k, harm_deg, harmonic_type_function,
                                            use_hanging_function, list_id_vertices_hanging,
                                            use_rational_function, rational_num_rat_points,
                                            list_id_vertices_rationals, rational_type_function,
                                            edge_normalization_square, hang_func)


    flags: Flags = set_flags(network_input_dimension,
                              method_order,
                              num_vertices,
                              num_hidden_layers,
                              num_neurons_per_layer,
                              harmonic_degree,
                              use_hanging_function,
                              list_id_vertices_rationals,
                              regularization_coefficient,
                              export_training_data_file_path)

    # Build the polynomial+harm correction. If no harmonic correction is requested,
    # all harmonic nn weights and biases are automatically set to zero
    nn = NAVEMNetwork(flags)
    nn.build_network()

    num_training_polygons = mesh.cell2_d_total_number()

    # Initialize edge loss
    tf.print('The number of points on each edge is: {}'.format(n_points_on_each_edge))
    bloss = losses.EdgesLoss(geometry_utilities, n_points_on_each_edge, vem_k, n_edges)

    points = np.array([], dtype=np.float64).reshape(0, network_input_dimension)
    angles = np.array([], dtype=np.float64).reshape(0, num_vertices)
    edges = np.array([], dtype=np.float64).reshape(0, num_vertices)
    polar_coords = np.array([], dtype=np.float64).reshape(0, 3 * num_vertices)

    tangents = np.array([], dtype=np.float64).reshape(0, 2)

    # training_polygons = np.array([], dtype=np.float64).reshape(0, 6)
    labels = np.array([], dtype=np.float64)
    labels_derivs = np.array([], dtype=np.float64)
    vertex_filter = np.array([], dtype=np.float64).reshape(0)

    super_vandermonde = np.zeros((num_training_polygons * num_vertices, num_points_on_each_edge * num_vertices, navem_polynomial.n_tot_polynomial))
    super_vander_dx = np.zeros((num_training_polygons * num_vertices, num_points_on_each_edge * num_vertices, navem_polynomial.n_tot_polynomial))
    super_vander_dy = np.zeros((num_training_polygons * num_vertices, num_points_on_each_edge * num_vertices, navem_polynomial.n_tot_polynomial))
    input_network = np.zeros((num_training_polygons * num_vertices, network_input_dimension))

    already_considered_polygons = []
    second_or_subsequent_polygon = False

    exact_bfgs = True
    for c in range(num_training_polygons):

        polygon = NAVEMPolygon(geometry_utilities, mesh_geometric_data.cell2_ds_vertices[c],
                               mesh_geometric_data.cell2_ds_diameters[c],
                               mesh_geometric_data.cell2_ds_centroids[c],
                               mesh_geometric_data.cell2_ds_triangulations[c])

        for v_id in range(num_vertices):

            rotated_vertices, _, _, scaling = polygon.map_f_inv(polygon.vertices, v_id)
            rotated_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)
            vertex_distance = scaling * np.roll(geometric_data[c].polygon.mapped_max_vertex_distance, shift=-v_id)

            c_points, c_labels, c_tangents, c_labels_derivs = bloss.add_polygon(rotated_vertices, add_coords=True)

            points = np.concatenate([points, c_points])

            labels = np.concatenate([labels, c_labels])
            labels_derivs = np.concatenate([labels_derivs, c_labels_derivs])
            tangents = np.concatenate([tangents, c_tangents])

            Npoint = c_points.shape[0]

            vander_points = tf.transpose(c_points).numpy()
            vander_points = vander_points[0:2, :]
            zeros = np.zeros([1, Npoint])
            vander_points = np.concatenate((vander_points, zeros), axis=0)

            # I store the vandermonde matrix that will be used
            if use_rational_function:
                polygon_edge_normals = geometry_utilities.compute_polygon_edge_normals(rotated_vertices)
                polygon_vertex_bisectors = geometry_utilities.compute_polygon_vertex_bisectors(polygon_edge_normals)
                polygon_bounding_box = geometry_utilities.compute_points_bounding_box(rotated_vertices)
                polygon_scale = max([polygon_bounding_box[0, 1] - polygon_bounding_box[0, 0],
                                     polygon_bounding_box[1, 1] - polygon_bounding_box[1, 0]])

                flat_poles, poles = RationalFunction.compute_poles(geometry_utilities, rational_num_rat_points,
                                                                   list_id_vertices_rationals, rotated_vertices,
                                                                   polygon_vertex_bisectors, polygon_scale)
                num_total_poles = flat_poles.shape[1]
                dist = RationalFunction.compute_dist(list_id_vertices_rationals, poles, flat_poles, rotated_vertices)

                local_vandermonde_grads = navem_polynomial.vander_derivatives(vander_points, rotated_vertices,
                                                                              flat_poles, dist, internal_angles=internal_angles,
                                                                              vertex_distance=vertex_distance)
                local_vander = navem_polynomial.vander(vander_points, rotated_vertices, flat_poles, dist, internal_angles=internal_angles,
                                                       vertex_distance=vertex_distance)
            else:
                local_vander, local_vandermonde_grads = (
                    navem_polynomial.vander_and_vander_derivatives(vander_points,
                                                                   rotated_vertices,
                                                                   internal_angles=internal_angles,
                                                                   vertex_distance=vertex_distance))

            super_vandermonde[c * num_vertices + v_id, :, :] = local_vander
            super_vander_dx[c * num_vertices + v_id, :, :] = local_vandermonde_grads[0, :, :]
            super_vander_dy[c * num_vertices + v_id, :, :] = local_vandermonde_grads[1, :, :]

            input_network[c * num_vertices + v_id, :] = rotated_vertices[:2, 1:].flatten()



    Npoint = points.shape[0]

    print("Global Max value vandermonde: ", np.max(abs(super_vandermonde)))

    print("\nNodi totali:", Npoint,
          "\nNodi totali filtrati:", sum(vertex_filter == 0),
          "\nNodi filtrati per un poligono (in media):", sum(vertex_filter == 0) / (num_training_polygons * n_vertices),
          "\nNodi filtrati attorno ad un vertice (in media):", sum(vertex_filter == 0) / (num_training_polygons * n_vertices**2))

    if Npoint == 0:
        raise ValueError("Not valid polygons")

    points = tf.convert_to_tensor(points, dtype=tf.float64)
    angles = tf.convert_to_tensor(angles, dtype=tf.float64)
    edges = tf.convert_to_tensor(edges, dtype=tf.float64)
    polar_coords = tf.convert_to_tensor(polar_coords, dtype=tf.float64)
    labels = tf.squeeze(tf.convert_to_tensor(labels, dtype=tf.float64))
    labels_derivs = tf.convert_to_tensor(labels_derivs, dtype=tf.float64)
    tangents = tf.convert_to_tensor(tangents, dtype=tf.float64)
    vertex_filter = tf.convert_to_tensor(vertex_filter, dtype=tf.float64)

    expCyclLr = trainlib.ExpCyclicLr(n_pol_epochs_order1, lr_min, lr_max)
    cb_list = [expCyclLr]

    if export_train_info:
        list_of_losses = []
        list_of_times = []
        storeLoss = trainlib.StoreLoss(list_of_losses, n_steps=1)
        storeTime = trainlib.StoreTime(list_of_times, n_steps=1)
        cb_list = [expCyclLr, storeLoss, storeTime]

    nn.nn_pol.deriv_labels.assign(tf.reshape(labels_derivs, (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
    nn.nn_pol.tangent_x.assign(tf.reshape(tangents[:, 0], (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
    nn.nn_pol.tangent_y.assign(tf.reshape(tangents[:, 1], (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
    nn.nn_pol.vertex_filter.assign(tf.reshape(vertex_filter, (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
    nn.nn_pol.train_vander_dx.assign(tf.convert_to_tensor(np.array(super_vander_dx), dtype=tf.float64))
    nn.nn_pol.train_vander_dy.assign(tf.convert_to_tensor(np.array(super_vander_dy), dtype=tf.float64))

    nn.nn_pol.train_vander.assign( tf.convert_to_tensor(super_vandermonde, dtype=tf.float64) )

    training_input = tf.convert_to_tensor(input_verts, dtype=tf.float64)
    labels = tf.reshape(labels, (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge))

    considered_loss = nn.nn_pol.copy_value_and_grad if use_grad_in_train else nn.nn_pol.copy_value
    tf.print("\n---------- I train the vem_k + harmonic polynomial part ----------\n")

    if split_deriv:
        tf.print("------------------- Copy of the basis functions -------------------\n")
        pol_results = trainlib.train_adam_bfgs(nn.nn_pol, considered_loss, training_input, labels, n_pol_epochs_order1,
                                               n_pol_epochs_order2, cb_list, use_bfgs=exact_bfgs)

        tf.print("\n---------- Copy of the basis functios derivatives ----------\n")
        nn.nn_pol_deriv.copy_weights(nn.nn_pol)

        nn.nn_pol_deriv.train_vander_dx.assign( tf.convert_to_tensor(np.array(super_vander_dx), dtype=tf.float64) )
        nn.nn_pol_deriv.train_vander_dy.assign( tf.convert_to_tensor(np.array(super_vander_dy), dtype=tf.float64) )
        nn.nn_pol_deriv.deriv_labels.assign(tf.reshape(labels_derivs, (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
        nn.nn_pol_deriv.tangent_x.assign(tf.reshape(tangents[:, 0], (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
        nn.nn_pol_deriv.tangent_y.assign(tf.reshape(tangents[:, 1], (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
        nn.nn_pol_deriv.vertex_filter.assign(tf.reshape(vertex_filter, (num_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
        considered_loss_dx = nn.nn_pol_deriv.copy_grad

        if export_train_info:
            list_of_losses_deriv = []
            list_of_times_deriv = []
            storeLoss = trainlib.StoreLoss(list_of_losses_deriv, n_steps=1)
            storeTime = trainlib.StoreTime(list_of_times_deriv, n_steps=1)
            cb_list = [expCyclLr, storeLoss, storeTime]

        pol_deriv_results = trainlib.train_adam_bfgs(nn.nn_pol_deriv, considered_loss_dx, training_input, labels,
                                                     n_pol_epochs_order1, n_pol_epochs_order2, cb_list,
                                                     use_bfgs=exact_bfgs)

    else:
        pol_results = trainlib.train_adam_bfgs(nn.nn_pol, considered_loss, training_input, labels, n_pol_epochs_order1,
                                               n_pol_epochs_order2, cb_list, use_bfgs=exact_bfgs)

    if export_train_info:
        adam_steps = np.arange(0, len(list_of_losses))
        bfgs_steps = np.arange(len(list_of_losses), len(list_of_losses)+len(pol_results[1].losses))

        steps = np.concatenate([adam_steps, bfgs_steps], axis=0)
        losses = np.concatenate([np.array(list_of_losses), np.array(pol_results[1].losses)], axis=0)
        times = np.concatenate([np.array(list_of_times), list_of_times[-1] + np.array(pol_results[1].times)], axis=0)

        fake_header = np.array([[n_pol_epochs_order1, len(pol_results[1].losses), 999999]])
        export_data = np.stack([steps, losses, times], axis=1)

        if split_deriv:
            adam_steps_deriv = np.arange(0, len(list_of_losses_deriv))
            bfgs_steps_deriv = np.arange(len(list_of_losses_deriv), len(list_of_losses_deriv)+len(pol_deriv_results[1].losses))

            steps_deriv = np.concatenate([adam_steps_deriv, bfgs_steps_deriv], axis=0)
            losses_deriv = np.concatenate([np.array(list_of_losses_deriv), np.array(pol_deriv_results[1].losses)], axis=0)
            times_deriv = np.concatenate([np.array(list_of_times_deriv), list_of_times_deriv[-1] + np.array(pol_deriv_results[1].times)], axis=0)
            export_data_deriv = np.stack([steps_deriv, losses_deriv, times_deriv], axis=1)
            data = np.concatenate([fake_header, export_data, export_data_deriv], axis=0)
        else:
            data = np.concatenate([fake_header, export_data], axis=0)

        np.savetxt('NeuralNetwork/savedModels/NAVEM_train_info.csv', data, delimiter='; ')

    # Storing the final loss and the l2 and h1 errors
    with open('{}_loss.csv'.format(FLAGS['name_storage']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['NPolygons', 'L2 loss', 'H1 loss'])

        pol_coeffs = nn.nn_pol.call(training_input)
        considered_loss(labels, pol_coeffs)
        if split_deriv:
            pol_coeffs_deriv = nn.nn_pol_deriv.call(training_input)
            considered_loss_dx(labels, pol_coeffs_deriv)

            writer.writerow([num_training_polygons,
                             np.sqrt(nn.nn_pol.curr_distance_l2.numpy()),
                             np.sqrt(nn.nn_pol_deriv.curr_distance_h1.numpy())])
        else:
            writer.writerow([num_training_polygons,
                             np.sqrt(nn.nn_pol.curr_distance_l2.numpy()),
                             np.sqrt(nn.nn_pol.curr_distance_h1.numpy())])
    # Final storage
    nn.saveModel()
