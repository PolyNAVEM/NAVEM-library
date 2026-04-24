from typing import TypedDict

class Flags(TypedDict):
    input_dim: int
    output_dim: int
    method_order: int
    method_type: int
    num_vertices: int
    mesh_import_path: str
    num_training_polygons: int
    num_hidden_layers: int
    num_neurons_per_layer: int
    num_epoches_opt_order1: int
    num_epoches_opt_order2: int
    learning_rate_max: float
    learning_rate_min: float
    use_sqrt_in_train: bool
    regularization_coefficient: float
    name_storage: str
    copy_basis_in_train: bool
    p_navem_exact_constant: bool
    quadrature_order: int
    distribution_points_type: int


def set_flags(network_input_dimension: int,
              method_order: int,
              method_type: int,
              num_vertices: int,
              mesh_import_path: str,
              num_training_polygons: int,
              num_hidden_layers: int,
              num_neurons_per_layer: int,
              num_epoches_opt_order1: int,
              num_epoches_opt_order2: int,
              learning_rate_max: float,
              learning_rate_min: float,
              use_sqrt_in_train: bool,
              regularization_coefficient: float,
              export_training_data_file_path: str,
              copy_basis_in_train: bool,
              p_navem_exact_constant: bool,
              quadrature_order: int,
              distribution_points_type: int) -> Flags:

    flags: Flags = {'input_dim': network_input_dimension,
                    'output_dim': 1,
                    'method_order': method_order,
                    'method_type': method_type,
                    'num_vertices': num_vertices,
                    'mesh_import_path': mesh_import_path,
                    'num_training_polygons': num_training_polygons,
                    'num_hidden_layers': num_hidden_layers,
                    'num_neurons_per_layer': num_neurons_per_layer,
                    'num_epoches_opt_order1': num_epoches_opt_order1,
                    'num_epoches_opt_order2': num_epoches_opt_order2,
                    'learning_rate_max': learning_rate_max,
                    'learning_rate_min': learning_rate_min,
                    'use_sqrt_in_train': use_sqrt_in_train,
                    'regularization_coefficient': regularization_coefficient,
                    'name_storage': export_training_data_file_path,
                    'copy_basis_in_train': copy_basis_in_train,
                    'p_navem_exact_constant': p_navem_exact_constant,
                    'quadrature_order': quadrature_order,
                    'distribution_points_type': distribution_points_type}

    return flags
