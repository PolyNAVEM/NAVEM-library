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

from NAVEM.NeuralNetwork.h_navem_network import Flags as H_Flags
from NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import Flags as BP_Flags
from NAVEM.PCC_2D.NAVEM_PCC_2D import NAVEMType
from NAVEM.Utilities.enforcing_boundary_functions import BubbleType, BoundaryMethodType
import tensorflow as tf


def print_training_information(flags: H_Flags | BP_Flags):
    tf.print('\n\x1b[6;30;42m' + 'Beginning of the training phase with the following characteristics.' + '\x1b[0m')

    tf.print('\n\x1b[6;30;42m' + 'Method' + '\x1b[0m')
    method_type = NAVEMType(flags["method_type"])
    tf.print("Method type: " + method_type.name)
    tf.print("Method order: " + str(flags["method_order"]))

    tf.print('\n\x1b[6;30;42m' + 'Training' + '\x1b[0m')
    tf.print("Training mesh path: " + flags["mesh_import_path"])
    tf.print("Number of steps of the first-order optimizer: " + str(flags["num_epoches_opt_order1"]))
    tf.print("Number of steps of the second-order optimizer: " + str(flags["num_epoches_opt_order2"]))
    tf.print("Regularization coefficient: " + str(flags["regularization_coefficient"]))

    tf.print('\n\x1b[6;30;42m' + 'Architecture' + '\x1b[0m')
    tf.print("Number of hidden layers: " + str(flags["num_hidden_layers"]))
    tf.print("Number of neurons per layer: " + str(flags["num_neurons_per_layer"]))

    match method_type:
        case method_type.H_NAVEM:
            tf.print('\n\x1b[6;30;42m' + method_type.name + ' properties' + '\x1b[0m')
            tf.print("Harmonic polynomials degree: " + str(flags["harmonic_degree"]))
            tf.print("Normalization diameter: " + str(flags["normalization_diameter"]))

        case method_type.B_NAVEM | method_type.P_NAVEM:
            boundary_method_type_g = BoundaryMethodType(flags["boundary_method_type_g"])
            boundary_method_type_adf = BoundaryMethodType(flags["boundary_method_type_adf"])
            bubble_type = BubbleType(flags["bubble_type"])

            tf.print('\n\x1b[6;30;42m' + method_type.name + ' properties' + '\x1b[0m')
            tf.print("Bubble function formula: " + bubble_type.name)
            tf.print("Bubble function type: " + boundary_method_type_adf.name)
            tf.print("Lifting function type: " + boundary_method_type_g.name)

        case _:
            raise ValueError("Not valid method type.")
