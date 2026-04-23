from pypolydim import gedim
from src.NAVEM.Utilities.NAVEM_PCC_2D import NAVEMType

def train_exact_bc_navem_pcc_2d_on_generic_polygon(method_order: int, method_type: NAVEMType,
                                                   num_vertices: int, mesh: gedim.MeshMatricesDAO,
                                                   num_hidden_layers: int, num_epoches_opt_order1: int,
                                                   num_epoches_opt_order2: int,
                                                   learning_rate_opt_order1: float, learning_rate_opt_order2: float,
                                                   regularization_coefficient: float,
                                                   export_training_data_file_path: str):

    assert method_order == 1


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

    FLAGS = {}
    FLAGS['input_dim'] = 2 * n_vertices
    FLAGS['output_dim'] = 1
    FLAGS['vem_order'] = vem_k
    FLAGS['n_edges'] = n_vertices

    FLAGS['n_hidden_layers'] = n_hidden_layers
    FLAGS['n_neurons'] = n_neurons
    FLAGS['net_reg_coef'] = net_reg_coef
    FLAGS['split_deriv'] = split_deriv
    FLAGS['use_grad'] = use_grad_in_train
    FLAGS['copy_pols'] = copy_pols_in_train
    FLAGS['napem_exact_one'] = napem_exact_one

    FLAGS['name_storage'] = ('NeuralNetwork/savedModels/{}_{}_k{}_nedges{}_arch_{}_{}_'
                             'tolAlmostHang{}_convex{}').format(
        method, name, vem_k, n_vertices, n_hidden_layers, n_neurons, int(tol_almost_hanging_nodes), concave_convex_both)

    # Network definition
    if method == "BNAVEM":
        nn = BCNetwork()
        nn.build_network(FLAGS)
    elif method == "NAPEM":
        nn = NapemSupernetwork(FLAGS)
    else:
        raise ValueError("You can only use BNAVEM or NAPEM")

    if fine_tune:
        if method == "BNAVEM":
            neural_file = ('./Navem2D/NetworkConfiguration2D/bnavem_models_2d/{}_{}_k{}_nedges{}_arch_{}_{}_'
                           'tolAlmostHang{}_convex{}.ckpt').format(
                "BNAVEM", name, vem_k, n_vertices, n_hidden_layers, n_neurons, int(tol_almost_hanging_nodes),
                concave_convex_both)

        elif method == "NAPEM":
            neural_file = ('./Navem2D/NetworkConfiguration2D/napem_models_2d/{}_{}_k{}_nedges{}_arch_{}_{}_'
                           'tolAlmostHang{}_convex{}.ckpt').format(
                "NAPEM", name, vem_k, n_vertices, n_hidden_layers, n_neurons, int(tol_almost_hanging_nodes),
                concave_convex_both)
        else:
            raise ValueError("not valid method")

        FLAGS['name_storage'] = FLAGS['name_storage'] + "_fineTuned"
        nn.load_weights(neural_file).expect_partial()

        n_epochs_order1 = 1
        lr_min = 1e-30
        lr_max = 1e-30

    file = open('{}_dictionary.txt'.format(FLAGS['name_storage']), 'w')
    file.write("network_input_dimension = {}\n".format(FLAGS['input_dim']))
    file.write("output_dim = {}\n".format(FLAGS['output_dim']))
    file.write("vem_k = {}\n".format(FLAGS['vem_order']))
    file.write("n_vertices = {}\n".format(FLAGS['n_edges']))
    file.write("n_edges = {}\n".format(n_vertices))
    file.write("n_training_polygons = {}\n".format(n_training_polygons))
    file.write("net_reg_coef = {}\n".format(net_reg_coef))
    if read_mesh:
        file.write("read_mesh = True\n")
    else:
        file.write("read_mesh = False\n")
    file.write("name_mesh = '{}'\n".format(name_mesh))
    file.write("tol_almost_hanging_nodes = {}\n".format(tol_almost_hanging_nodes))
    file.write("use_almost_hanging = {}\n".format(use_almost_hanging))
    file.write("name = '{}'\n".format(name))
    file.write("n_hidden_layers = {}\n".format(FLAGS['n_hidden_layers']))
    file.write("n_neurons = {}\n".format(FLAGS['n_neurons']))
    file.write("n_epochs_order1 = {}\n".format(n_epochs_order1))
    file.write("n_epochs_order2 = {}\n".format(n_epochs_order2))
    file.write("pts_type = '{}'\n".format(pts_type))
    file.write("quadrature_order = {}\n".format(quadrature_order))
    file.write("lr_max = {}\n".format(lr_min))
    file.write("lr_min = {}\n".format(lr_min))
    file.write("concave_convex_both = {}\n".format(concave_convex_both))
    file.write("split_deriv = {}\n".format(split_deriv))
    file.write("method = '{}'\n".format(method))
    if copy_pols_in_train:
        file.write("copy_pols_in_train = True\n")
    else:
        file.write("copy_pols_in_train = False\n")
    if use_grad_in_train:
        file.write("use_grad_in_train = True\n")
    else:
        file.write("use_grad_in_train = False\n")
    if napem_exact_one:
        file.write("napem_exact_one = True\n")
    else:
        file.write("napem_exact_one = False\n")

    file.close()

    # Define the training set
    if read_mesh:

        if name_mesh == "convex_concave" or name_mesh == "structured_concave":
            domain_vertices = np.empty([3, 4])
            domain_vertices[0] = [0.0, 1.0, 1.0, 0.0]
            domain_vertices[1] = [0.0, 0.0, 1.0, 1.0]
            domain_vertices[2] = [0.0, 0.0, 0.0, 0.0]

            rectangle_origin = domain_vertices[:, 0]
            rectangle_base_tangent = domain_vertices[:, 1] - domain_vertices[:, 0]
            rectangle_height_tangent = domain_vertices[:, 3] - domain_vertices[:, 0]

            if name_mesh == "convex_concave":
                train_mesh = mesh_generators.create_convex_concave_mesh(rectangle_origin, rectangle_base_tangent,
                                                                        rectangle_height_tangent,
                                                                        2, 2)
            elif name_mesh == "structured_concave":
                train_mesh = mesh_generators.create_structured_concave_mesh(rectangle_origin, rectangle_base_tangent,
                                                                            rectangle_height_tangent,
                                                                            1, 1)
        else:
            train_mesh = mesh_utilities.import_mesh_from_csv(name_mesh)
    elif concave_convex_both == 0:
        pol_generator = PolygonGenerator(geometry_utilities)
        num_concave_polygons = n_training_polygons
        train_mesh = pol_generator.generate_polygons(0, num_concave_polygons, 1,
                                                     n_vertices, use_almost_hanging)
    elif n_training_polygons == 1 and n_edges == 4:
        train_mesh = mesh_square(geometry_utilities)
    elif concave_convex_both == 1:
        pol_generator = PolygonGenerator(geometry_utilities)
        num_convex_polygons = n_training_polygons
        train_mesh = pol_generator.generate_polygons(num_convex_polygons, 0, 1,
                                                     n_vertices, use_almost_hanging)
    elif concave_convex_both == 2:
        pol_generator = PolygonGenerator(geometry_utilities)
        num_convex_polygons = int(n_training_polygons/2)
        num_concave_polygons = n_training_polygons - num_convex_polygons
        train_mesh = pol_generator.generate_polygons(num_convex_polygons, num_concave_polygons, 1,
                                                     n_vertices, use_almost_hanging)

    path = "./ExportParaview/Training_Polygon"
    # Check whether the specified path exists or not
    is_exist = os.path.exists(path)
    if is_exist:
        import shutil

        shutil.rmtree(path)
        os.makedirs(path)
        print("The new directory is created!")
    else:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

    csv_path = FLAGS['name_storage']
    # Check whether the specified path exists or not
    is_exist = os.path.exists(csv_path)
    if is_exist:
        import shutil

        shutil.rmtree(csv_path)
        os.makedirs(csv_path)
    else:
        # Create a new directory because it does not exist
        os.makedirs(csv_path)

    mesh_utilities.export_mesh_to_csv(train_mesh, csv_path)

    # Actual number of polygons
    geometric_data = mesh_utilities.compute_geometric_properties_mesh_2(geometry_utilities, train_mesh)

    concave_counter = 0
    if read_mesh:
        list_element = []
        tmp_n_training_polygons = 0
        num_elements = len(train_mesh.cells_2)
        for c in range(num_elements):
            if len(train_mesh.cells_2[c].vertices) == n_vertices:
                # Select convex or concave polygons
                if concave_convex_both == 1 and not geometric_data[c].mapped_polygon_is_convex:
                    continue
                if concave_convex_both == 0 and geometric_data[c].mapped_polygon_is_convex:
                    continue
                if (concave_convex_both == 2 and
                        concave_counter >= n_training_polygons / 2 and
                        not geometric_data[c].mapped_polygon_is_convex):  # already more than half concave
                    continue

                if (concave_convex_both == 2 and
                        tmp_n_training_polygons - concave_counter >= n_training_polygons / 2 and
                        geometric_data[c].mapped_polygon_is_convex):  # already more than half convex
                    continue

                if not geometric_data[c].mapped_polygon_is_convex:
                    concave_counter += 1

                list_element.append(c)
                tmp_n_training_polygons += 1
                if tmp_n_training_polygons == n_training_polygons:
                    break
        n_training_polygons = tmp_n_training_polygons
    else:
        list_element = [c for c in range(n_training_polygons)]

    (num_total_element, min_num_vertices, avg_num_vertices, max_num_vertices,
     min_area, avg_area, max_area, min_diameter, avg_diameter, max_diameter,
     min_edge_ratio, avg_edge_ratio, max_edge_ratio,
     min_anisotropic_ratio, avg_anisotropic_ratio, max_anisotropic_ratio) \
        = mesh_utilities.reset_mesh_statistics()

    (num_total_element, min_num_vertices, avg_num_vertices, max_num_vertices,
     min_area, avg_area, max_area, min_diameter, avg_diameter, max_diameter,
     min_edge_ratio, avg_edge_ratio, max_edge_ratio,
     min_anisotropic_ratio, avg_anisotropic_ratio, max_anisotropic_ratio) \
        = mesh_utilities.compute_mesh_geometry_statistics(geometry_utilities, train_mesh, geometric_data,
                                                          num_total_element, min_num_vertices,
                                                          avg_num_vertices, max_num_vertices,
                                                          min_area, avg_area, max_area,
                                                          min_diameter, avg_diameter, max_diameter,
                                                          min_edge_ratio, avg_edge_ratio, max_edge_ratio,
                                                          min_anisotropic_ratio, avg_anisotropic_ratio,
                                                          max_anisotropic_ratio)

    mesh_utilities.export_mesh_geometry_statistics(csv_path, 'training_mesh',
                                                   num_total_element, min_num_vertices,
                                                   avg_num_vertices, max_num_vertices,
                                                   min_area, avg_area, max_area,
                                                   min_diameter, avg_diameter, max_diameter,
                                                   min_edge_ratio, avg_edge_ratio, max_edge_ratio,
                                                   min_anisotropic_ratio, avg_anisotropic_ratio,
                                                   max_anisotropic_ratio)

    tf.print('\n')
    tf.print('The actual number of polygons is: {}'.format(n_training_polygons))
    tf.print(
        'Convex polygons: {}. \t Concave polygons: {}'.format(n_training_polygons - concave_counter, concave_counter))
    tf.print('\n')

    if pts_type == "gauss_triangle":
        quadrature = QuadratureGauss2DTriangle(quadrature_order)
        ref_nodes, ref_weights = quadrature.fill_points_and_weights()
        n_points_per_pol = ref_weights.shape[0] * n_vertices

    else:
        nodes_generator = TrianglesGrid()
        if pts_type == "uniform":
            border_pts = False
            if border_pts:
                n_points_per_pol = int((quadrature_order + 1) * (quadrature_order + 0) * 0.5) * n_vertices + 1
            else:
                n_points_per_pol = int((quadrature_order - 1) * (quadrature_order - 2) * 0.5) * n_vertices
        elif pts_type == "gauss_polygon":
            border_type = 1
            ref_nodes = nodes_generator.concentrating_grid_over_triangle(quadrature_order, 1, border_type)
            n_points_per_pol = ref_nodes.shape[1] * n_vertices
            if border_type == 0:
                n_points_per_pol -= quadrature_order * n_vertices + (n_vertices - 1)
        else:
            raise ValueError("The pts-type variable should be uniform, gauss_triangle or gauss_polygon!")

    n_functions_per_pol = n_vertices - napem_exact_one

    inputs = np.zeros((n_training_polygons * n_points_per_pol * n_functions_per_pol, network_input_dimension))
    training_quad_w = np.zeros((n_training_polygons * n_points_per_pol * n_functions_per_pol, 1))

    xy_per_pol = np.zeros(shape=(n_training_polygons, n_points_per_pol, 2))
    vertices_per_pol = np.zeros(shape=(n_training_polygons, 2, n_vertices))
    jac_per_pol = np.zeros(shape=(n_training_polygons, 2, 2, n_functions_per_pol))
    jac_inv_per_pol = np.zeros(shape=(n_training_polygons, 2, 2, n_functions_per_pol))

    exact_bfgs = True

    for i in range(n_training_polygons):
        c = list_element[i]
        vertices_idxs = train_mesh.cells_2[c].vertices
        hanging_vertices = geometric_data[c].id_aligned_vertices
        internal_point = train_mesh.cells_2[c].internal_point
        cell_vertices = geometric_data[c].vertex_points
        diameter = geometric_data[c].diameter
        centroid = geometric_data[c].centroid
        polygon = geometric_data[c].polygon

        # I use this polygon to train only if it has the current number of vertices
        # (if we work with meshes with different typologies of polygons,
        # not all the polygons are to be taken into account!)
        if len(vertices_idxs) != n_vertices:
            continue

        # # Select convex or concave polygons
        # if not train_on_concave and not geometric_data[c].mapped_polygon_is_convex:
        #     tf.print(c, ' is not convex')
        #     continue
        #
        # if train_on_concave and geometric_data[c].mapped_polygon_is_convex:
        #     tf.print(c, ' is convex')
        #     continue

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

        xy_per_pol[i, :, :] = inertia_mapped_internal_nodes[:2, :].T
        vertices_per_pol[i, :, :] = polygon.mapped_vertices[:2, :]

        for v_id in range(n_functions_per_pol):
            rotated_vertices, resc_mapped_points, jac_inv, jac, inv_rescaling_factor = map_fix_vertex_inv(
                polygon.mapped_vertices,
                inertia_mapped_internal_nodes,
                v_id)
            rotated_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)

            flat_rotated_vertices = rotated_vertices[:2, 1:].flatten()
            rep_flat_rot_vertices = np.tile(flat_rotated_vertices[np.newaxis, :], [n_points_per_pol, 1])
            c_inputs = np.concatenate([resc_mapped_points[:2, :].T, rep_flat_rot_vertices], axis=1)

            inputs[(i * n_functions_per_pol + v_id) * n_points_per_pol: (
                                                                                i * n_functions_per_pol + v_id + 1) * n_points_per_pol,
            :] = c_inputs
            # training_quad_w[(i*n_functions_per_pol + v_id) * n_points_per_pol : (i*n_functions_per_pol + v_id+1) * n_points_per_pol, 0] = inertia_mapped_internal_weights * inv_rescaling_factor
            jac_per_pol[i, :, :, v_id] = jac[:2, :2]
            jac_inv_per_pol[i, :, :, v_id] = jac_inv[:2, :2]

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
    # training_quad_w = tf.convert_to_tensor(training_quad_w, dtype=tf.float64)
    # nn.training_quad_w.assign(training_quad_w)
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

    # print(xy_per_pol.shape, vertices_per_pol.shape, jac_per_pol.shape)
    # aa

    nn.setupModel_global_input(xy_per_pol, vertices_per_pol, jac_per_pol, setup_n_derivatives,
                               geometry_utilities)

    expCyclLr = trainlib.ExpCyclicLr(n_epochs_order1, lr_min, lr_max)
    cb_list = [expCyclLr]

    if export_train_info:
        list_of_losses = []
        list_of_times = []
        storeLoss = trainlib.StoreLoss(list_of_losses, n_steps=1)
        storeTime = trainlib.StoreTime(list_of_times, n_steps=1)
        cb_list = [expCyclLr, storeLoss, storeTime]

    tf.print("\n---------- Start training ----------\n")

    if method == "BNAVEM":
        del vertices_per_pol
        del jac_per_pol
        del jac_inv_per_pol
        del xy_per_pol
        del geometric_data
        del train_mesh

        results = trainlib.train_adam_bfgs(nn, considered_loss, inputs, labels, n_epochs_order1,
                                           n_epochs_order2, cb_list, use_bfgs=exact_bfgs)
    else:
        results = trainlib.train_adam_bfgs(nn.nn_func, considered_loss, inputs, labels, n_epochs_order1,
                                           n_epochs_order2, cb_list, use_bfgs=exact_bfgs)

        if split_deriv:
            tf.print("\n---------- Start gradient training ----------\n")
            func_pred_grad = nn.nn_func.get_u_and_du(inputs)[:, 1:]
            nn.nn_grad.store_terms_for_loss(func_pred_grad, vertices_per_pol, jac_inv_per_pol)
            expCyclLr = trainlib.ExpCyclicLr(n_epochs_order1, lr_min, lr_max)
            cb_list = [expCyclLr]

            if export_train_info:
                list_of_losses_deriv = []
                list_of_times_deriv = []
                storeLoss = trainlib.StoreLoss(list_of_losses_deriv, n_steps=1)
                storeTime = trainlib.StoreTime(list_of_times_deriv, n_steps=1)
                cb_list = [expCyclLr, storeLoss, storeTime]

            results2 = trainlib.train_adam_bfgs(nn.nn_grad, nn.nn_grad.inter_grad_loss, inputs, labels, n_epochs_order1,
                                                n_epochs_order2, cb_list, use_bfgs=exact_bfgs)

    if export_train_info:
        adam_steps = np.arange(0, len(list_of_losses))
        bfgs_steps = np.arange(len(list_of_losses), len(list_of_losses)+len(results[1].losses))

        steps = np.concatenate([adam_steps, bfgs_steps], axis=0)
        losses = np.concatenate([np.array(list_of_losses), np.array(results[1].losses)], axis=0)
        times = np.concatenate([np.array(list_of_times), list_of_times[-1] + np.array(results[1].times)], axis=0)

        fake_header = np.array([[n_epochs_order1, len(results[1].losses), 999999]])
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
    with open('{}_loss.csv'.format(FLAGS['name_storage']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['NPolygons', 'Laplacian loss'])

        nn_output = nn.call(inputs)
        considered_loss(labels, nn_output)
        if method == "BNAVEM":
            writer.writerow([n_training_polygons,
                             np.sqrt(nn.curr_laplacian.numpy())])
        elif method == "NAPEM":
            writer.writerow([n_training_polygons,
                             nn.nn_func.loss_one.numpy(),
                             nn.nn_func.loss_x.numpy(),
                             nn.nn_func.loss_y.numpy()])
    # Final storage
    nn.saveModel()

    ending_time = time.perf_counter()
    print("Total execution time:", ending_time - starting_time)
