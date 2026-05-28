from NAVEM.NeuralNetwork.h_navem_network import Flags as H_Flags
from NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import Flags as BP_Flags
from NAVEM.PCC_2D.NAVEM_PCC_2D import NAVEMType
from NAVEM.Utilities.enforcing_boundary_functions import BubbleType, BoundaryMethodType


def print_training_information(flags: H_Flags | BP_Flags):
    print('\n\x1b[6;30;42m' + 'Beginning of the training phase with the following characteristics.' + '\x1b[0m')

    print('\n\x1b[6;30;42m' + 'Method' + '\x1b[0m')
    method_type = NAVEMType(flags["method_type"])
    print("Method type: " + method_type.name)
    print("Method order: " + str(flags["method_order"]))

    print('\n\x1b[6;30;42m' + 'Training' + '\x1b[0m')
    print("Training mesh path: " + flags["mesh_import_path"])
    print("Number of steps of the first-order optimizer: " + str(flags["num_epoches_opt_order1"]))
    print("Number of steps of the second-order optimizer: " + str(flags["num_epoches_opt_order2"]))
    print("Regularization coefficient: " + str(flags["regularization_coefficient"]))

    print('\n\x1b[6;30;42m' + 'Architecture' + '\x1b[0m')
    print("Number of hidden layers: " + str(flags["num_hidden_layers"]))
    print("Number of neurons per layer: " + str(flags["num_neurons_per_layer"]))

    match method_type:
        case method_type.H_NAVEM:
            print('\n\x1b[6;30;42m' + method_type.name + ' properties' + '\x1b[0m')
            print("Harmonic polynomials degree: " + str(flags["harmonic_degree"]))
            print("Normalization diameter: " + str(flags["normalization_diameter"]))

        case method_type.B_NAVEM | method_type.P_NAVEM:
            boundary_method_type_g = BoundaryMethodType(flags["boundary_method_type_g"])
            boundary_method_type_adf = BoundaryMethodType(flags["boundary_method_type_adf"])
            bubble_type = BubbleType(flags["bubble_type"])

            print('\n\x1b[6;30;42m' + method_type.name + ' properties' + '\x1b[0m')
            print("Bubble function formula: " + bubble_type.name)
            print("Bubble function type: " + boundary_method_type_adf.name)
            print("Lifting function type: " + boundary_method_type_g.name)

        case _:
            raise ValueError("Not valid method type.")
