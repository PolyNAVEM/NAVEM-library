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

import cProfile
import os.path
from pathlib import Path
import argparse
from pypolydim import gedim, polydim
from NAVEM.NeuralNetwork.train_generic_pcc_2d import train_navem_pcc_2d_on_generic_polygon
from NAVEM.NeuralNetwork.train_generic_exact_bc_pcc_2d import train_exact_bc_navem_pcc_2d_on_generic_polygon
from NAVEM.PCC_2D.NAVEM_PCC_2D import NAVEMType, NAVEMElementType
from NAVEM.geometry.geometry_utilities import compute_geometric_properties_mesh_2

import os

from NAVEM.Utilities.enforcing_boundary_functions import BoundaryMethodType, BubbleType

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf


def main():
    program_folder = str(Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser()
    parser.add_argument('-order', '--method-order', dest='method_order',
                        default=1, type=int, help="Method order (Default: 1)")
    parser.add_argument('-method', '--method-type', dest='method_type',
                        default=1, type=int, help="Method type: 1 - H-NAVEM; 2 - B-NAVEM; 3 - P-NAVEM (Default: 1)")
    parser.add_argument('-mesh', '--mesh-type', dest='mesh_type', default=4,
                        type=int, help="Mesh generator type: 3 - OFFImporter; 4 - CSV CsvImporter (Default: 4)")
    parser.add_argument('-import', '--import-path', dest='import_path',
                        default=program_folder + '/../TrainingDataset/TrainingReferenceSquare', type=str,
                        help="Mesh Import Path")
    parser.add_argument('-e', '--num-vertices', dest='num_vertices', default=4, type=int,
                        help='Number of vertices of the polygons (Default: 4)')
    parser.add_argument('-tol1', '--tolerance-1-d', dest='tolerance1_d', default=1.0e-12,
                        type=float, help="Geometric Tolerance 1D")
    parser.add_argument('-tol2', '--tolerance-2-d', dest='tolerance2_d', default=1.0e-14,
                        type=float, help="Geometric Tolerance 2D")
    parser.add_argument('-et', '--element-type', dest='element_type', default=1,
                        type=int, help='Train the neural network using polygons that are: 1 - generic convex; '
                                       '2 - generic concave')

    # Network
    parser.add_argument('-n', '--export-training-data-file-path', dest='export_training_data_file_path',
                        default=program_folder + '/../Export/TrainedModels/', help='Storage path of the outputs')
    parser.add_argument('-nhl', '--num-hidden-layers', dest='num_hidden_layers', default=3, type=int,
                        help='Number of hidden layers for the polynomial neural network')
    parser.add_argument('-nnl', '--num-neurons-per-layer', dest='num_neurons_per_layer', default=30, type=int,
                        help='Number of nodes per hidden layers for the polynomial neural network')

    parser.add_argument('-neo1p', '--num-epoches-opt-order1', dest='num_epoches_opt_order1', default=100, type=int,
                        help='Number of training epochs with first order optimizer')
    parser.add_argument('-neo2p', '--num-epoches-opt-order2', dest='num_epoches_opt_order2', default=50, type=int,

                        help='Number of training epochs with second order optimizer')
    parser.add_argument('-lr_max', '--lr-max', dest='learning_rate_max', default=1e-2, type=float,
                        help='Maximum value of the learning rate')
    parser.add_argument('-lr_min', '--lr-min', dest='learning_rate_min', default=1e-3, type=float,
                        help='Minimum value of the learning rate')
    parser.add_argument('-rc', '--regularization-coefficient', dest='regularization_coefficient', default=1e-8,
                        type=float,
                        help='Regularization coefficient')
    parser.add_argument('-usit', '--use-sqrt-in-train', dest='use_sqrt_in_train', default=1, type=int,
                        help='Flag to specify if the NNs are trained using also the sqrt in loss definition')
    parser.add_argument('-eti', '--export-train-info', dest='export_training_info', default=False,
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='Flag to specify if I want to export info (loss, time,...) during training')

    # NAVEM
    parser.add_argument('-np', '--num-points-on-edges', dest='num_points_on_each_edge', default=50, type=int,
                        help='Number of points on each edge')
    parser.add_argument('-nsd', '--normalization-diameter', dest='normalization_diameter', default=5.0,
                        type=float, help='Edge of the normalization square')
    parser.add_argument('-hd', '--harmonic-degree', dest='harmonic_degree', default=10, type=int,
                        help='Degree of the considered harmonic polynomials')
    parser.add_argument('-hf', '--use-hanging-function', dest='use_hanging_function', default=1, type=int,
                        help='Flag to specify to add the hanging function in the linear combination')

    # B-NAVEM and P-NAVEM
    parser.add_argument('-qo', '--quadrature-order', dest='quadrature_order', default=10, type=int,
                        help='Order of the quadrature rule to generate the training points')
    parser.add_argument('-pt', '--distribution-points-type', dest='distribution_points_type', type=int, default=3,
                        help='Type of points: 1 - gauss; 2 - uniform; 3 - accumulated (Default: 3)')
    parser.add_argument('-cbt', '--copy-basis-in-train', dest='copy_basis_in_train', default=False,
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='Flag to specify if distance from basis functions should be added to the loss')
    parser.add_argument('-bt', '--bubble-type', dest='bubble_type', type=int, default=1,
                        help='Type of points: 1 - approximate_distance_function; 2 - product (Default: 1)')
    parser.add_argument('-bmt', '--boundary-method-type', dest='boundary_method_type', default=2, type=int,
                        help='Boundary method type: 1 - line; 2 - segment (Default: 2)')

    args = parser.parse_args()

    method_type = NAVEMType(args.method_type)
    element_type = NAVEMElementType(args.element_type)

    export_file_path = args.export_training_data_file_path
    if not os.path.exists(export_file_path):
        os.makedirs(export_file_path)

    export_file_path = args.export_training_data_file_path + "/" + method_type.name
    if not os.path.exists(export_file_path):
        os.makedirs(export_file_path)

    tf.keras.backend.set_floatx('float64')
    tf.keras.backend.set_epsilon(args.tolerance1_d)

    pr = cProfile.Profile()
    pr.enable()

    geometry_utilities_config = gedim.GeometryUtilitiesConfig()
    geometry_utilities_config.tolerance1_d = args.tolerance1_d
    geometry_utilities_config.tolerance2_d = args.tolerance2_d
    geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
    mesh_utilities = gedim.MeshUtilities()

    mesh_data = gedim.MeshMatrices()
    mesh = gedim.MeshMatricesDAO(mesh_data)

    polydim.pde_tools.mesh.pde_mesh_utilities.import_mesh_2_d(geometry_utilities,
                                                              mesh_utilities,
                                                              polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D(
                                                                  args.mesh_type),
                                                              args.import_path,
                                                              mesh)

    mesh_geometric_data = compute_geometric_properties_mesh_2(geometry_utilities, mesh_utilities, mesh)

    full_export_training_data_file_path = (export_file_path+ "/num_vertices_" + str(args.num_vertices) + "_" + element_type.name + "/")
    if not os.path.exists(full_export_training_data_file_path):
        os.makedirs(full_export_training_data_file_path)

    match method_type:
        case method_type.H_NAVEM:
            train_navem_pcc_2d_on_generic_polygon(args.method_order,
                                                  method_type,
                                                  args.num_vertices,
                                                  geometry_utilities,
                                                  mesh,
                                                  mesh_geometric_data,
                                                  args.import_path,
                                                  args.num_hidden_layers,
                                                  args.num_neurons_per_layer,
                                                  args.num_epoches_opt_order1,
                                                  args.num_epoches_opt_order2,
                                                  args.learning_rate_max,
                                                  args.learning_rate_min,
                                                  args.num_points_on_each_edge,
                                                  args.regularization_coefficient,
                                                  full_export_training_data_file_path,
                                                  args.export_training_info,
                                                  args.use_sqrt_in_train,
                                                  element_type,
                                                  args.harmonic_degree,
                                                  args.normalization_diameter,
                                                  args.use_hanging_function)
        case method_type.B_NAVEM | method_type.P_NAVEM:
            train_exact_bc_navem_pcc_2d_on_generic_polygon(args.method_order,
                                                           method_type,
                                                           args.num_vertices,
                                                           geometry_utilities,
                                                           mesh,
                                                           mesh_geometric_data,
                                                           args.import_path,
                                                           args.num_hidden_layers,
                                                           args.num_neurons_per_layer,
                                                           args.num_epoches_opt_order1,
                                                           args.num_epoches_opt_order2,
                                                           args.learning_rate_max,
                                                           args.learning_rate_min,
                                                           args.quadrature_order,
                                                           args.distribution_points_type,
                                                           args.regularization_coefficient,
                                                           full_export_training_data_file_path,
                                                           args.export_training_info,
                                                           args.copy_basis_in_train,
                                                           args.use_sqrt_in_train,
                                                           element_type,
                                                           BoundaryMethodType(args.boundary_method_type),
                                                           BubbleType(args.bubble_type))
        case _:
            raise ValueError("not valid method")

    pr.disable()
    pr.dump_stats(full_export_training_data_file_path + "/program.prof")


if __name__ == '__main__':
    main()
