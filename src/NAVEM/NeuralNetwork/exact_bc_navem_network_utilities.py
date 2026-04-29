from typing import TypedDict, Dict

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
                    'quadrature_order': quadrature_order,
                    'distribution_points_type': distribution_points_type}

    return flags


def write_flags_on_dictionary(flags: Flags) -> None:

    file = open('{}/dictionary.txt'.format(flags['name_storage']), 'w')
    file.write("method_type = {}\n".format(flags['method_type']))
    file.write("method_order = {}\n".format(flags['method_order']))
    file.write("input_dim = {}\n".format(flags['input_dim']))
    file.write("output_dim = {}\n".format(flags['output_dim']))
    file.write("num_vertices = {}\n".format(flags['num_vertices']))
    file.write("num_training_polygons = {}\n".format(flags['num_training_polygons']))
    file.write("regularization_coefficient = {}\n".format(flags['regularization_coefficient']))
    file.write("mesh_import_path = {}\n".format(flags['mesh_import_path']))
    file.write("quadrature_order = {}\n".format(flags['quadrature_order']))
    file.write("distribution_points_type = {}\n".format(flags['distribution_points_type']))
    file.write("num_hidden_layers = {}\n".format(flags['num_hidden_layers']))
    file.write("num_neurons_per_layer = {}\n".format(flags['num_neurons_per_layer']))
    file.write("num_epoches_opt_order1 = {}\n".format(flags['num_epoches_opt_order1']))
    file.write("num_epoches_opt_order2 = {}\n".format(flags['num_epoches_opt_order2']))
    file.write("learning_rate_max = {}\n".format(flags['learning_rate_max']))
    file.write("learning_rate_min = {}\n".format(flags['learning_rate_min']))
    file.write("use_sqrt_in_train = {}\n".format(flags['use_sqrt_in_train']))
    file.write("copy_basis_in_train = {}\n".format(flags['copy_basis_in_train']))

    file.close()

def load_flags_from_dictionary(name_storage: str, raw: Dict) -> Flags:

    flags: Flags = {'input_dim': int(raw["input_dim"]),
                    'output_dim': int(raw["output_dim"]),
                    'method_order': int(raw["method_order"]),
                    'method_type': int(raw["method_type"]),
                    'num_vertices': int(raw["num_vertices"]),
                    'mesh_import_path': raw["mesh_import_path"],
                    'num_training_polygons': int(raw["num_training_polygons"]),
                    'num_hidden_layers': int(raw["num_hidden_layers"]),
                    'num_neurons_per_layer': int(raw["num_neurons_per_layer"]),
                    'num_epoches_opt_order1': int(raw["num_epoches_opt_order1"]),
                    'num_epoches_opt_order2': int(raw["num_epoches_opt_order2"]),
                    'learning_rate_max': float(raw["learning_rate_max"]),
                    'learning_rate_min': float(raw["learning_rate_min"]),
                    'use_sqrt_in_train': bool(raw["use_sqrt_in_train"]),
                    'regularization_coefficient': float(raw["regularization_coefficient"]),
                    'name_storage': name_storage,
                    'copy_basis_in_train': bool(raw["copy_basis_in_train"]),
                    'quadrature_order': int(raw["quadrature_order"]),
                    'distribution_points_type': int(raw["distribution_points_type"])}

    return flags