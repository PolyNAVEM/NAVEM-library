# -*- coding: utf-8 -*-
from geometry import mesh_utilities, mesh_generators
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
from networkTraining import losses
from NeuralNetwork.pol_and_harm_network import PolHarmNetwork
import networkTraining.training_utils as trainlib
from geometry.geometry_properties import GeometryUtilities
import csv
from geometry.polygon_generators import PolygonGenerator
from NNVEMUtilities.NAVEMPolynomialBasis import NAVEMPolynomialBasis
from NNVEMUtilities.RationalFunction import RationalFunction
from NNVEMUtilities.LaplaceSolver import LaplaceSolver
from geometry.custom_meshes import mesh_square
import os
import time
from VTKUtilities.VTK_utilities import export_polygon


if __name__ == '__main__':

    starting_time = time.perf_counter()

    # Workarounds to try to push tensorflow hidden computations to double precision
    K.set_floatx('float64')
    tf.keras.backend.set_floatx('float64')
    K.set_epsilon(1e-13)
    # Here we fix the random seed to test on the same polygon each time the script is launched
    # THIS IS FALSE
    # tf.random.set_seed(0)
    # tf.keras.utils.set_random_seed(0)
    # np.random.seed(0)

    # Reading from command line all required information
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--vemk', dest='vemk', default=1, type=int, help='VEM order of accuracy k')
    parser.add_argument('-e', '--nvertices', dest='nvertices', default=4, type=int,
                        help='Number of vertices of the polygon(s)')
    parser.add_argument('-rm', '--read_mesh', dest='read_mesh', default=0, type=int,
                        help='Read training mesh from input')
    parser.add_argument('-tm', '--training_mesh', dest='training_mesh', default='./Mesh/square/square4x4',
                        help='Path for training mesh')
    parser.add_argument('-p', '--npolygons', dest='npolygons', default=100, type=int,
                        help='Number of polygons used to train')
    parser.add_argument('-np', '--npointsonedges', dest='n_points_on_each_edge', default=50, type=int,
                        help='Number of points on each edge')

    parser.add_argument('-n', '--name', dest='name', default='test', help='Storage name of the neural network')
    parser.add_argument('-nphl', '--nhiddenlayers-polnn', dest='n_pol_hidden_layers', default=3, type=int,
                        help='Number of hidden layers for the polynomial neural network')
    parser.add_argument('-npnhl', '--nnodes-hiddenlayers-polnn', dest='n_pol_neurons', default=30, type=int,
                        help='Number of nodes per hidden layers for the polynomial neural network')
    parser.add_argument('-neo1p', '--nepochs-order1-polnn', dest='n_pol_epochs_order1', default=1000, type=int,
                        help='Number of training epochs with first order optimizer polinomial NN')
    parser.add_argument('-neo2p', '--nepochs-order2-polnn', dest='n_pol_epochs_order2', default=500, type=int,
                        help='Number of training epochs with second order optimizer polinomial NN')
    parser.add_argument('-lrmax', '--lrmax', dest='lr_max', default=1e-3, type=float,
                        help='Maximum value of the learning rate')
    parser.add_argument('-lrmin', '--lrmin', dest='lr_min', default=1e-3, type=float,
                        help='Minimum value of the learning rate')

    parser.add_argument('-tah', '--tol-almost-hanging', dest='tol_almost_hanging', default=0.1, type=float,
                        help='Flag to specify tolerance for almost hanging nodes')
    parser.add_argument('-ah', '--use-almost-hanging', dest='use_almost_hanging', default=0, type=int,
                        help='Flag to specify if the almost hanging nodes must be considered or not')

    parser.add_argument('--concave-convex-both', dest='concave_convex_both', default=0, type=int,
                        help='Train only on concave polygons, only on convex polygons, or on boh')
    parser.add_argument('-regc', '--regolarization-coeff', dest='net_reg_coef', default=1e-8, type=float,
                        help='Regolarization coefficient')
    parser.add_argument('-nsq', '--edge-normalization-square', dest='edge_normalization_square', default=5.0,
                        type=float, help='Edge of the normalization square')

    parser.add_argument('-harmdeg', '--harmonic-degree', dest='harm_deg', default=10, type=int,
                        help='Degree of the considered harmonic polynomials')
    parser.add_argument('-harmtype', '--harmonic-type-function', dest='harmonic_type_function', default='total',
                        help='Type of the considered harmonic polynomials (total, real or imag)')

    parser.add_argument('-rf', '--use-rationalFunction', dest='use_rationalFunc', default=0, type=int,
                        help='Flag to specify to add the hanging function in the linear combination')
    parser.add_argument('-ratnrp', '--rational-number-of-rat-pts', dest='rational_num_rat_points', default=10, type=int,
                        help='Number of points for the rational functions')
    parser.add_argument('-rattype', '--rational-type-function', dest='rational_type_function', default='total',
                        help='Type of the considered rational (total, real or imag)')

    parser.add_argument('-hf', '--use-hangingFunction', dest='use_hangingFunc', default=0, type=int,
                        help='Flag to specify to add the hanging function in the linear combination')

    parser.add_argument('-sd', '--split-derivatives', dest='split_deriv', default=1, type=int,
                        help='Flag to specify if the polynomial derivatives are represented by separate NNs')
    parser.add_argument('-ugit', '--use-grad-in-train', dest='use_grad_in_train', default=0, type=int,
                        help='Flag to specify if the NNs are trained using also the gradient')
    parser.add_argument('-usit', '--use-sqrt-in-train', dest='use_sqrt_in_train', default=0, type=int,
                        help='Flag to specify if the NNs are trained using also the sqrt in loss definition')
    parser.add_argument('-eti', '--export-train-info', dest='export_train_info', default=1, type=int,
                        help='Flag to specify if I want to export info (loss, time,...) during training')


    args = parser.parse_args()
    vem_k = args.vemk
    n_vertices = args.nvertices
    n_edges = n_vertices
    n_training_polygons = args.npolygons
    read_mesh = args.read_mesh == 1
    concave_convex_both = args.concave_convex_both
    name_mesh = args.training_mesh
    n_points_on_each_edge = args.n_points_on_each_edge

    tol_almost_hanging_nodes = args.tol_almost_hanging
    use_almost_hanging = args.use_almost_hanging == 1

    name = args.name
    n_pol_hidden_layers = args.n_pol_hidden_layers
    n_pol_neurons = args.n_pol_neurons
    n_pol_epochs_order1 = args.n_pol_epochs_order1
    n_pol_epochs_order2 = args.n_pol_epochs_order2
    lr_max = args.lr_max
    lr_min = args.lr_min
    net_reg_coef = args.net_reg_coef
    edge_normalization_square = args.edge_normalization_square

    # grado per armonici
    harm_deg = args.harm_deg
    harmonic_type_function = args.harmonic_type_function

    use_hanging_function = args.use_hangingFunc == 1

    use_rational_function = args.use_rationalFunc == 1
    rational_num_rat_points = args.rational_num_rat_points if use_rational_function else 0
    rational_type_function = args.rational_type_function if use_rational_function else 0

    split_deriv = args.split_deriv == 1
    use_grad_in_train = args.use_grad_in_train == 1
    use_sqrt_in_train = args.use_sqrt_in_train == 1

    export_train_info = args.export_train_info == 1

    # Here we print a bunch of checks and useful information for debugging and backward check
    # These information will be printed in the corresponding log file
    tf.print('\n')
    tf.print('Method: NAVEM')
    tf.print('Selected VEM order: {}'.format(vem_k))
    tf.print('Selected number of vertices: {}'.format(n_vertices))
    tf.print('Selected number of training polygons: {}'.format(n_training_polygons))
    tf.print('Selected number of hidden layers polynomial NN: {}'.format(n_pol_hidden_layers))
    tf.print('Selected number of nodes per hidden layers polynomial NN: {}'.format(n_pol_neurons))
    tf.print('Selected number of training epochs for the first order optimizer for the polynomial NN: {}'.format(
        n_pol_epochs_order1))
    tf.print('Selected number of training epochs for the second order optimizer for the polynomial NN: {}'.format(
        n_pol_epochs_order2))
    tf.print(
        'The training starts with a learning rate of {} up to a minimum acceptable value of {}'.format(lr_max, lr_min))
    tf.print('The base name of the total NN is {}'.format(name))
    tf.print('Degree of the harmonic polynomials is {}'.format(harm_deg))
    tf.print('Will I use the Hanging function? {}'.format(use_hanging_function))

    tf.print('Will I split the derivatives of the linear part? {}'.format(split_deriv))
    tf.print('Will I use the gradient during the training? {}'.format(use_grad_in_train))
    tf.print('Will I use the sqrt during the training? {}'.format(use_sqrt_in_train))
    tf.print('Regularization coefficient {}'.format(net_reg_coef))
    tf.print('Edge of the normalization square {}'.format(edge_normalization_square))
    tf.print('Concave, convex or both? {}'.format(concave_convex_both))
    tf.print('\n')

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

    tf.print('\n')
    tf.print('The total number of polynomials is: {}'.format(navem_polynomial.n_tot_polynomial))
    tf.print('\n')

    FLAGS = {}
    FLAGS['input_dim'] = network_input_dimension
    FLAGS['output_dim'] = 1
    FLAGS['vem_order'] = vem_k
    FLAGS['n_edges'] = n_vertices

    FLAGS['n_pol_hidden_layers'] = n_pol_hidden_layers
    FLAGS['n_pol_neurons'] = n_pol_neurons

    FLAGS['harm_deg'] = harm_deg
    FLAGS['harmonic_type_function'] = harmonic_type_function

    FLAGS['use_hanging_function'] = use_hanging_function
    if use_hanging_function:
        FLAGS['hanging_harm_deg'] = navem_polynomial.hanging_function.harm_deg
        FLAGS['hanging_num_rat_points'] = navem_polynomial.hanging_function.flat_poles.shape[1]
        FLAGS['hanging_type_rational'] = navem_polynomial.hanging_function.function_type
        FLAGS['hanging_type_harmonic'] = navem_polynomial.hanging_function.function_type
        FLAGS['list_id_vertices_hanging'] = list_id_vertices_hanging
    else:
        hanging_harm_deg = 0
        hanging_num_rat_points = 0
        hanging_type_rational = 'real'
        hanging_type_harmonic = 'real'

    FLAGS['use_rational_function'] = use_rational_function
    FLAGS['list_id_vertices_rationals'] = list_id_vertices_rationals
    if use_rational_function:
        FLAGS['rational_num_rat_points'] = rational_num_rat_points
        FLAGS['rational_type_function'] = rational_type_function
        FLAGS['list_id_vertices_rationals'] = list_id_vertices_rationals
    else:
        rational_num_rat_points = 0
        rational_type_function = 'real'

    FLAGS['num_tot_polynomials'] = navem_polynomial.n_tot_polynomial

    FLAGS['split_deriv'] = split_deriv
    FLAGS['use_grad_in_train'] = use_grad_in_train
    FLAGS['use_sqrt_in_train'] = use_sqrt_in_train
    FLAGS['net_reg_coef'] = net_reg_coef

    FLAGS['name_storage'] = ('NeuralNetwork/savedModels/{}_k{}_nedges{}_arch_{}_{}_harm_{}_{}_'
                             'useHangFunc{}_'
                             'useRatFunc{}_rat_{}_{}_'
                             'splitDeriv{}_useGradInTrain{}_useSqrtInTrain{}_normalSquare{}_'
                             'convex{}').format(
        name, vem_k, n_vertices, n_pol_hidden_layers, n_pol_neurons, harm_deg, harmonic_type_function,
        int(use_hanging_function),
        int(use_rational_function), rational_num_rat_points, rational_type_function,
        int(split_deriv), int(use_grad_in_train),int(use_sqrt_in_train), edge_normalization_square,
        concave_convex_both)

    # Build the polynomial+harm correction. If no harmonic correction is requested,
    # all harmonic nn weights and biases are automatically set to zero
    nn = PolHarmNetwork()
    nn.build_network(FLAGS)

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
        num_convex_polygons = 0
        num_concave_polygons = n_training_polygons
        train_mesh = pol_generator.generate_polygons(0, num_concave_polygons, 1,
                                                     n_vertices, use_almost_hanging)
    elif n_training_polygons == 1 and n_edges == 4:
        train_mesh = mesh_square(geometry_utilities)
    elif concave_convex_both == 1:
        pol_generator = PolygonGenerator(geometry_utilities)
        num_convex_polygons = n_training_polygons
        num_concave_polygons = 0
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
    if read_mesh:
        tf.print('Convex polygons: {}. \t Concave polygons: {}'.format(n_training_polygons - concave_counter,
                                                                       concave_counter))
    else:
        tf.print('Convex polygons: {}. \t Concave polygons: {}'.format(num_convex_polygons, num_concave_polygons))
    tf.print('\n')

    # Initialize edge loss
    tf.print('The number of points on each edge is: {}'.format(n_points_on_each_edge))
    bloss = losses.EdgesLoss(geometry_utilities, n_points_on_each_edge, vem_k, n_edges)

    points = np.array([], dtype=np.float64).reshape(0, network_input_dimension)
    angles = np.array([], dtype=np.float64).reshape(0, n_vertices)
    edges = np.array([], dtype=np.float64).reshape(0, n_vertices)
    polar_coords = np.array([], dtype=np.float64).reshape(0, 3 * n_vertices)

    tangents = np.array([], dtype=np.float64).reshape(0, 2)

    # training_polygons = np.array([], dtype=np.float64).reshape(0, 6)
    labels = np.array([], dtype=np.float64)
    labels_derivs = np.array([], dtype=np.float64)
    vertex_filter = np.array([], dtype=np.float64).reshape(0)

    super_vandermonde = np.zeros((n_training_polygons*n_vertices, n_points_on_each_edge*n_vertices, navem_polynomial.n_tot_polynomial))
    super_vander_dx = np.zeros((n_training_polygons*n_vertices, n_points_on_each_edge*n_vertices, navem_polynomial.n_tot_polynomial))
    super_vander_dy = np.zeros((n_training_polygons*n_vertices, n_points_on_each_edge*n_vertices, navem_polynomial.n_tot_polynomial))
    input_verts = np.zeros((n_training_polygons * n_vertices, 2 * (n_vertices - 1)))

    already_considered_polygons = []
    second_or_subsequent_polygon = False

    exact_bfgs = True
    for i in range(len(list_element)):

        c = list_element[i]
        vertices_idxs = train_mesh.cells_2[c].vertices
        hanging_vertices = geometric_data[c].id_aligned_vertices
        internal_point = train_mesh.cells_2[c].internal_point
        cell_vertices = geometric_data[c].vertex_points
        diameter = geometric_data[c].diameter
        centroid = geometric_data[c].centroid


        polygon = geometric_data[c].polygon
        mapped_angles = geometric_data[c].mapped_polygon_internal_angles
        mapped_angles = np.array(mapped_angles)

        # I use this polygon to train only if it has the current number of vertices
        # (if we work with meshes with different typologies of polygons,
        # not all the polygons are to be taken into account!)
        if len(vertices_idxs) != n_vertices:
            continue

        for v_id in range(n_vertices):

            rotated_vertices, _, _, scaling = polygon.map_f_inv(polygon.vertices, v_id)
            rotated_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)
            vertex_distance = scaling * np.roll(geometric_data[c].polygon.mapped_max_vertex_distance, shift=-v_id)

            # if i >= 845 and i <= 850:
            #     export_polygon(path + "/polygon_" + str(n_vertices * c + v_id), rotated_vertices)

            # # uncomment to avoid considering the same basis function 1354513457987 times
            # if second_or_subsequent_polygon:
            #     if np.any(np.all(np.linalg.norm(rotated_vertices - already_considered_polygons, ord=1, axis=0) < tol)):
            #         continue
            #     else:
            #         already_considered_polygons = np.concatenate([already_considered_polygons,
            #                                                       np.expand_dims(rotated_vertices, 0)], axis=0)
            # else:
            #     second_or_subsequent_polygon = True
            #     already_considered_polygons = np.expand_dims(rotated_vertices, 0)

            #new_input_polygon = np.expand_dims(rotated_vertices[:2, 1:].flatten(), 0)
            c_points, c_labels, c_tangents, c_labels_derivs = bloss.add_polygon(rotated_vertices, add_coords=True)

            internal_angles = np.roll(mapped_angles, axis=1, shift=-v_id)
            c_angles = np.tile(internal_angles, [c_points.shape[0], 1])

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

            super_vandermonde[i*n_vertices + v_id, :, :] = local_vander
            super_vander_dx[i*n_vertices + v_id, :, :] = local_vandermonde_grads[0, :, :]
            super_vander_dy[i*n_vertices + v_id, :, :] = local_vandermonde_grads[1, :, :]

            # print("Max value vandermonde: ", np.max(abs(local_vander)), " c = ", n_vertices * c + v_id)

            input_verts[i * n_vertices + v_id, :] = rotated_vertices[:2, 1:].flatten()

            local_vertex_filter = np.ones((Npoint,))
            for vi in range(cell_vertices.shape[1]):
                local_vertex_filter *= np.linalg.norm(c_points[:, :2] - rotated_vertices[:2, vi], axis=1) > 4e-2
            vertex_filter = np.concatenate([vertex_filter, local_vertex_filter])

    Npoint = points.shape[0]

    print("Global Max value vandermonde: ", np.max(abs(super_vandermonde)))

    print("\nNodi totali:", Npoint,
          "\nNodi totali filtrati:", sum(vertex_filter == 0),
          "\nNodi filtrati per un poligono (in media):", sum(vertex_filter == 0) / (n_training_polygons * n_vertices),
          "\nNodi filtrati attorno ad un vertice (in media):", sum(vertex_filter == 0) / (n_training_polygons * n_vertices**2))

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

    nn.nn_pol.deriv_labels.assign(tf.reshape(labels_derivs, (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
    nn.nn_pol.tangent_x.assign(tf.reshape(tangents[:, 0], (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
    nn.nn_pol.tangent_y.assign(tf.reshape(tangents[:, 1], (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
    nn.nn_pol.vertex_filter.assign(tf.reshape(vertex_filter, (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
    nn.nn_pol.train_vander_dx.assign(tf.convert_to_tensor(np.array(super_vander_dx), dtype=tf.float64))
    nn.nn_pol.train_vander_dy.assign(tf.convert_to_tensor(np.array(super_vander_dy), dtype=tf.float64))

    nn.nn_pol.train_vander.assign( tf.convert_to_tensor(super_vandermonde, dtype=tf.float64) )

    training_input = tf.convert_to_tensor(input_verts, dtype=tf.float64)
    labels = tf.reshape(labels, (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge))

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
        nn.nn_pol_deriv.deriv_labels.assign(tf.reshape(labels_derivs, (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
        nn.nn_pol_deriv.tangent_x.assign(tf.reshape(tangents[:, 0], (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
        nn.nn_pol_deriv.tangent_y.assign(tf.reshape(tangents[:, 1], (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
        nn.nn_pol_deriv.vertex_filter.assign(tf.reshape(vertex_filter, (n_training_polygons * n_vertices, n_vertices * n_points_on_each_edge)))
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

            writer.writerow([n_training_polygons,
                             np.sqrt(nn.nn_pol.curr_distance_l2.numpy()),
                             np.sqrt(nn.nn_pol_deriv.curr_distance_h1.numpy())])
        else:
            writer.writerow([n_training_polygons,
                             np.sqrt(nn.nn_pol.curr_distance_l2.numpy()),
                             np.sqrt(nn.nn_pol.curr_distance_h1.numpy())])
    # Final storage
    nn.saveModel()

    ending_time = time.perf_counter()
    print("Total execution time:", ending_time - starting_time)
