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
from warnings import catch_warnings

import numpy as np
from numpy.typing import NDArray
from typing import List, TypedDict, Tuple, Dict
from pypolydim import gedim, polydim
from NAVEM.Utilities.points_generator import map_pts_from_1d_to_2d

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.layers import Dense


class Flags(TypedDict):
    input_dim: int
    output_dim: int
    method_order: int
    method_type: int
    basis_function_type: int
    element_type: int
    num_vertices: int
    num_generators: int
    mesh_import_path: str
    num_training_polygons: int
    num_hidden_layers: int
    num_neurons_per_layer: int
    num_epoches_opt_order1: int
    num_epoches_opt_order2: int
    learning_rate_max: float
    learning_rate_min: float
    use_sqrt_in_train: bool
    harmonic_degree: int
    normalization_diameter: float
    use_hanging_function: bool
    list_id_vertices_hanging: List[int]
    num_rationals_points: int
    rational_type_function: int
    list_id_vertices_rationals: List[int]
    regularization_coefficient: float
    num_points_on_each_edge: int
    name_storage: str


def set_flags(network_input_dimension: int,
              method_order: int,
              method_type: int,
              basis_function_type: int,
              element_type: int,
              num_vertices: int,
              num_generators: int,
              mesh_import_path: str,
              num_training_polygons: int,
              num_hidden_layers: int,
              num_neurons_per_layer: int,
              num_epoches_opt_order1: int,
              num_epoches_opt_order2: int,
              learning_rate_max: float,
              learning_rate_min: float,
              use_sqrt_in_train: bool,
              harmonic_degree: int,
              normalization_diameter: float,
              use_hanging_function: bool,
              list_id_vertices_hanging: List[int],
              num_rationals_points: int,
              rational_type_function: int,
              list_id_vertices_rationals: List[int],
              regularization_coefficient: float,
              num_points_on_each_edge: int,
              export_training_data_file_path: str) -> Flags:
    flags: Flags = {'input_dim': network_input_dimension,
                    'output_dim': 1,
                    'method_order': method_order,
                    'method_type': method_type,
                    'basis_function_type': basis_function_type,
                    'element_type': element_type,
                    'num_vertices': num_vertices,
                    'num_generators': num_generators,
                    'mesh_import_path': mesh_import_path,
                    'num_training_polygons': num_training_polygons,
                    'num_hidden_layers': num_hidden_layers,
                    'num_neurons_per_layer': num_neurons_per_layer,
                    'num_epoches_opt_order1': num_epoches_opt_order1,
                    'num_epoches_opt_order2': num_epoches_opt_order2,
                    'learning_rate_max': learning_rate_max,
                    'learning_rate_min': learning_rate_min,
                    'use_sqrt_in_train': use_sqrt_in_train,
                    'harmonic_degree': harmonic_degree,
                    'normalization_diameter': normalization_diameter,
                    'use_hanging_function': use_hanging_function,
                    'list_id_vertices_hanging': list_id_vertices_hanging,
                    'num_rationals_points': num_rationals_points,
                    'rational_type_function': rational_type_function,
                    'list_id_vertices_rationals': list_id_vertices_rationals,
                    'regularization_coefficient': regularization_coefficient,
                    'num_points_on_each_edge': num_points_on_each_edge,
                    'name_storage': export_training_data_file_path}

    return flags


def write_flags_on_dictionary(flags: Flags) -> None:
    file = open('{}/dictionary.txt'.format(flags['name_storage']), 'w')
    file.write("method_type = {}\n".format(flags['method_type']))
    file.write("basis_function_type = {}\n".format(flags['basis_function_type']))
    file.write("method_order = {}\n".format(flags['method_order']))
    file.write("element_type = {}\n".format(flags['element_type']))
    file.write("input_dim = {}\n".format(flags['input_dim']))
    file.write("output_dim = {}\n".format(flags['output_dim']))
    file.write("num_vertices = {}\n".format(flags['num_vertices']))
    file.write("num_training_polygons = {}\n".format(flags['num_training_polygons']))
    file.write("regularization_coefficient = {}\n".format(flags['regularization_coefficient']))
    file.write("mesh_import_path = {}\n".format(flags['mesh_import_path']))
    file.write("num_points_on_each_edge = {}\n".format(flags['num_points_on_each_edge']))

    file.write("num_hidden_layers = {}\n".format(flags['num_hidden_layers']))
    file.write("num_neurons_per_layer = {}\n".format(flags['num_neurons_per_layer']))
    file.write("num_epoches_opt_order1 = {}\n".format(flags['num_epoches_opt_order1']))
    file.write("num_epoches_opt_order2 = {}\n".format(flags['num_epoches_opt_order2']))
    file.write("learning_rate_max = {}\n".format(flags['learning_rate_max']))
    file.write("learning_rate_min = {}\n".format(flags['learning_rate_min']))

    file.write("harmonic_degree = {}\n".format(flags['harmonic_degree']))
    file.write("normalization_diameter = {}\n".format(flags['normalization_diameter']))
    file.write("num_generators = {}\n".format(flags['num_generators']))
    file.write("use_hanging_function = {}\n".format(flags['use_hanging_function']))
    file.write("list_id_vertices_hanging = {}\n".format(flags['list_id_vertices_hanging']))
    file.write("num_rationals_points = {}\n".format(flags['num_rationals_points']))
    file.write("rational_type_function = {}\n".format(flags['rational_type_function']))
    file.write("list_id_vertices_rationals = {}\n".format(flags['list_id_vertices_rationals']))
    file.write("use_sqrt_in_train = {}\n".format(flags['use_sqrt_in_train']))

    file.close()


def load_flags_from_dictionary(name_storage: str, raw: Dict) -> Flags:

    try:
        list_id_vertices_hanging = [int(x.strip()) for x in raw["list_id_vertices_hanging"][1:-1].split(",")]
    except ValueError:
        list_id_vertices_hanging = []

    try:
        list_id_vertices_rationals = [int(x.strip()) for x in raw["list_id_vertices_rationals"][1:-1].split(",")]
    except ValueError:
        list_id_vertices_rationals = []

    flags: Flags = {
        "input_dim": int(raw["input_dim"]),
        "output_dim": int(raw["output_dim"]),
        "method_order": int(raw["method_order"]),
        "method_type": int(raw["method_type"]),
        "basis_function_type": int(raw["basis_function_type"]),
        "element_type": int(raw["element_type"]),
        "num_vertices": int(raw["num_vertices"]),
        "num_generators": int(raw["num_generators"]),
        "mesh_import_path": raw["mesh_import_path"],
        "num_training_polygons": int(raw["num_training_polygons"]),
        "num_hidden_layers": int(raw["num_hidden_layers"]),
        "num_neurons_per_layer": int(raw["num_neurons_per_layer"]),
        "num_epoches_opt_order1": int(raw["num_epoches_opt_order1"]),
        "num_epoches_opt_order2": int(raw["num_epoches_opt_order2"]),
        "learning_rate_max": float(raw["learning_rate_max"]),
        "learning_rate_min": float(raw["learning_rate_min"]),
        "use_sqrt_in_train": bool(raw["use_sqrt_in_train"]),
        "harmonic_degree": int(raw["harmonic_degree"]),
        "normalization_diameter": float(raw["normalization_diameter"]),
        "use_hanging_function": bool(raw["use_hanging_function"]),
        "list_id_vertices_hanging": list_id_vertices_hanging,
        "num_rationals_points": int(raw["num_rationals_points"]),
        "rational_type_function": int(raw["rational_type_function"]),
        "list_id_vertices_rationals": list_id_vertices_rationals,
        "regularization_coefficient": float(raw["regularization_coefficient"]),
        "num_points_on_each_edge": int(raw["num_points_on_each_edge"]),
        "name_storage": name_storage,
    }

    return flags


class BoundaryLoss:
    n_pts: int
    num_vertices: int
    reference_eval_points: NDArray[np.float64]
    basis_evaluations: NDArray[np.float64]
    derivatives_evaluations: NDArray[np.float64]

    def __init__(self, geometry_utilities: gedim.GeometryUtilities, method_order: int):

        self.geometry_utilities = geometry_utilities
        self.method_order = method_order

        assert method_order == 1

        lobatto_quadrature = gedim.quadrature.Quadrature_GaussLobatto1D()
        quadrature_data = lobatto_quadrature.fill_points_and_weights(self.method_order)
        self.references_do_fs_on_edge = quadrature_data.points[0, :]

        self.lagrange_coefficients = polydim.interpolation.lagrange.lagrange_1_d_coefficients(
            self.references_do_fs_on_edge)

    def initialize_basis_evaluations_on_boundary(self, reference_evaluation_points: NDArray[np.float64],
                                                 num_vertices: int, v_id: int = 0) -> None:

        assert reference_evaluation_points.ndim == 1

        self.n_pts = len(reference_evaluation_points)
        self.reference_eval_points = reference_evaluation_points

        self.num_vertices = num_vertices

        lagrange_values = polydim.interpolation.lagrange.lagrange_1_d_values(self.references_do_fs_on_edge,
                                                                             self.lagrange_coefficients,
                                                                             reference_evaluation_points)
        lagrange_derivatives_values = (
            polydim.interpolation.lagrange.lagrange_1_d_derivative_values(self.references_do_fs_on_edge,
                                                                          self.lagrange_coefficients,
                                                                          reference_evaluation_points))

        self.basis_evaluations = np.zeros([self.n_pts * self.num_vertices, 1])
        self.derivatives_evaluations = np.zeros([self.n_pts * self.num_vertices, 1])
        zero_pol = np.zeros(self.n_pts)

        # do_fs on vertices (evaluations)
        v = v_id
        prev_v = v_id - 1 if v_id > 0 else self.num_vertices - 1
        for e in range(self.num_vertices):
            if e == v:  # edge e after vertex v
                self.basis_evaluations[e * self.n_pts:(e + 1) * self.n_pts, 0] = lagrange_values[:, 0]
                self.derivatives_evaluations[e * self.n_pts:(e + 1) * self.n_pts, 0] = lagrange_derivatives_values[:, 0]
            elif e == prev_v:
                self.basis_evaluations[e * self.n_pts:(e + 1) * self.n_pts, 0] = lagrange_values[:, 1]
                self.derivatives_evaluations[e * self.n_pts:(e + 1) * self.n_pts, 0] = lagrange_derivatives_values[:, 1]
            else:
                self.basis_evaluations[e * self.n_pts:(e + 1) * self.n_pts, 0] = zero_pol
                self.derivatives_evaluations[e * self.n_pts:(e + 1) * self.n_pts, 0] = zero_pol

    def add_polygon(self, vertices: NDArray[np.float64], add_coordinates: bool = False) \
            -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

        nodes = np.zeros((2, self.num_vertices * self.n_pts))
        tangents = np.zeros((2, self.num_vertices * self.n_pts))

        # mapping the training pts from the reference element to the current one
        for e in range(self.num_vertices):
            v_start = np.expand_dims(vertices[:, e], 1)
            v_end = np.expand_dims(vertices[:, (e + 1) % self.num_vertices], 1)
            # tf.print("vertices",v_start, v_end)
            first_idx, last_idx = [e * self.n_pts, (e + 1) * self.n_pts]
            nodes[:, first_idx:last_idx] = map_pts_from_1d_to_2d(self.reference_eval_points, v_start, v_end)[:2, :]
            tangents[:, first_idx:last_idx] = (v_end - v_start)[:2, :]

        nodes = nodes.T
        edge_lengths = np.sqrt(tangents[0, :] ** 2 + tangents[1, :] ** 2)
        tangents /= edge_lengths  # tangent normalization
        # values of polynomials for boundary do_fs
        all_pols = np.squeeze(self.basis_evaluations.T.reshape(-1, 1))
        # values of derivatives for boundary do_fs
        all_derivatives = np.squeeze(self.derivatives_evaluations.T.reshape(-1, 1)) / edge_lengths

        if add_coordinates:
            xy_vertices = np.expand_dims(vertices[:2, 1:].flatten(), 0)
            rep_xy_vertices = np.repeat(xy_vertices, nodes.shape[0], axis=0)
            nodes = np.concatenate([nodes, rep_xy_vertices], axis=1)

        return nodes, all_pols, tangents.T, all_derivatives


class HNAVEMNeuralNetwork(tf.keras.Model):
    def __init__(self, input_dim: int, n_outputs: int, n_neurons: int, n_hidden_layers: int, reg_coefficient: float,
                 use_sqrt: bool):
        super(HNAVEMNeuralNetwork, self).__init__(name='h_navem_neural_network')
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.n_outputs = n_outputs
        self.reg_coefficient = reg_coefficient
        self.use_sqrt = use_sqrt

        self.layers_list = [
            Dense(self.n_neurons, activation='tanh', kernel_initializer='glorot_normal')]
        for _ in range(self.n_hidden_layers):
            self.layers_list.append(
                Dense(self.n_neurons, activation='tanh', kernel_initializer='glorot_normal'))
        self.layers_list.append(Dense(self.n_outputs, kernel_initializer='glorot_normal'))

        self.best_loss = tf.Variable(tf.convert_to_tensor(np.inf, dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.best_w = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False,
                                  shape=(None,), dtype=tf.float64)
        self.bfgs_regularization_grads = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False,
                                                     validate_shape=False, shape=(None,),
                                                     dtype=tf.float64)

        self.tangent_x = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                     validate_shape=False,
                                     shape=(None, None), dtype=tf.float64)
        self.tangent_y = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                     validate_shape=False,
                                     shape=(None, None), dtype=tf.float64)
        self.deriv_labels = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                        validate_shape=False,
                                        shape=(None, None), dtype=tf.float64)
        self.vertex_filter = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                         validate_shape=False,
                                         shape=(None, None), dtype=tf.float64)

        self.curr_distance_l2 = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                            validate_shape=False, dtype=tf.float64)
        self.curr_distance_h1 = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                            validate_shape=False, dtype=tf.float64)
        self.train_vander = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                        validate_shape=False, shape=(None, None, None), dtype=tf.float64)
        self.train_vander_dx = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None, None), dtype=tf.float64)
        self.train_vander_dy = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None, None), dtype=tf.float64)

    def build(self, input_shape):
        self.layers_list[0].build(input_shape)
        for layer in self.layers_list[1:]:
            layer.build((None, self.n_neurons))

    @staticmethod
    def apply_vandermonde(x, vander):
        return tf.einsum('pi,pji->pj', x, vander)

    def call(self, x):
        x = self.layers_list[0](x)
        for layer in self.layers_list[1:-1]:
            x = layer(x) + x
        return self.layers_list[-1](x)

    def copy_value_and_grad(self, y_true, y_predicted):
        value_cost = self.copy_value(y_true, y_predicted)
        deriv_cost = self.copy_grad(y_true, y_predicted)

        return value_cost + tf.square(deriv_cost)

    def copy_value(self, y_true, y_predicted):
        u = self.apply_vandermonde(y_predicted, self.train_vander)
        value_cost = tf.reduce_mean(tf.square(u - tf.squeeze(y_true)))
        self.curr_distance_l2.assign(value_cost)

        if self.use_sqrt:
            return tf.sqrt(value_cost)
        else:
            return value_cost

    def copy_grad(self, _y_true, y_predicted):
        ux = self.apply_vandermonde(y_predicted, self.train_vander_dx)
        uy = self.apply_vandermonde(y_predicted, self.train_vander_dy)
        tangent_derivative = ux * self.tangent_x + uy * self.tangent_y
        deriv_cost = tf.reduce_mean(tf.square(tangent_derivative - self.deriv_labels) * self.vertex_filter)
        self.curr_distance_h1.assign(deriv_cost)

        if self.use_sqrt:
            return tf.sqrt(deriv_cost)
        else:
            return deriv_cost

    def copy_weights(self, nn):
        if not self.built:
            self.build(input_shape=(None, self.input_dim))
        for i in range(len(self.layers_list)):
            self.layers_list[i].set_weights(nn.layers_list[i].get_weights())


class HNAVEMNetworksContainer:
    def __init__(self, flags: Flags):
        self.flags: Flags = flags

        self.nn_basis_function = HNAVEMNeuralNetwork(self.flags['input_dim'] - 2,
                                                     self.flags['num_generators'],
                                                     self.flags['num_neurons_per_layer'],
                                                     self.flags['num_hidden_layers'],
                                                     self.flags['regularization_coefficient'],
                                                     self.flags['use_sqrt_in_train'])

        self.nn_basis_derivatives = HNAVEMNeuralNetwork(self.flags['input_dim'] - 2,
                                                        self.flags['num_generators'],
                                                        self.flags['num_neurons_per_layer'],
                                                        self.flags['num_hidden_layers'],
                                                        self.flags['regularization_coefficient'],
                                                        self.flags['use_sqrt_in_train'])

    def save_model(self):
        zero_1d = tf.convert_to_tensor([0.], dtype=tf.float64)
        zero_2d = tf.convert_to_tensor([[0.]], dtype=tf.float64)
        zero_3d = tf.convert_to_tensor([[[0.]]], dtype=tf.float64)

        self.nn_basis_function.best_w.assign(zero_1d)
        self.nn_basis_function.bfgs_regularization_grads.assign(zero_1d)
        self.nn_basis_function.tangent_x.assign(zero_2d)
        self.nn_basis_function.tangent_y.assign(zero_2d)
        self.nn_basis_function.deriv_labels.assign(zero_2d)
        self.nn_basis_function.vertex_filter.assign(zero_2d)
        self.nn_basis_function.train_vander.assign(zero_3d)
        self.nn_basis_function.train_vander_dx.assign(zero_3d)
        self.nn_basis_function.train_vander_dy.assign(zero_3d)

        self.nn_basis_derivatives.best_w.assign(zero_1d)
        self.nn_basis_derivatives.bfgs_regularization_grads.assign(zero_1d)
        self.nn_basis_derivatives.tangent_x.assign(zero_2d)
        self.nn_basis_derivatives.tangent_y.assign(zero_2d)
        self.nn_basis_derivatives.deriv_labels.assign(zero_2d)
        self.nn_basis_derivatives.vertex_filter.assign(zero_2d)
        self.nn_basis_derivatives.train_vander.assign(zero_3d)
        self.nn_basis_derivatives.train_vander_dx.assign(zero_3d)
        self.nn_basis_derivatives.train_vander_dy.assign(zero_3d)

        self.nn_basis_function.save_weights(self.flags['name_storage'] + "/nn_function_weights.weights.h5")
        self.nn_basis_derivatives.save_weights(self.flags['name_storage'] + "/nn_derivatives_weights.weights.h5")

    def load_weights(self, name_storage: str):
        self.nn_basis_function.build(input_shape=(None, self.nn_basis_function.input_dim))
        self.nn_basis_derivatives.build(input_shape=(None, self.nn_basis_function.input_dim))
        self.nn_basis_function.load_weights(name_storage + "/nn_function_weights.weights.h5")
        self.nn_basis_derivatives.load_weights(name_storage + "/nn_derivatives_weights.weights.h5")
