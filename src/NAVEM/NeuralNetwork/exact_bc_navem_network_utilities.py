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

from typing import TypedDict, Dict
from abc import ABC, abstractmethod
import tensorflow as tf
from NAVEM.Utilities.enforcing_boundary_functions import EnforcingBoundary, BoundaryMethodType, BubbleType
from enum import Enum
from pypolydim import gedim
from numpy.typing import NDArray
import numpy as np


class SetupDerivatives(Enum):
    basis = 0
    basis_and_derivatives = 1
    basis_and_derivatives_and_laplacian = 2


class Flags(TypedDict):
    input_dim: int
    output_dim: int
    method_order: int
    method_type: int
    basis_function_type: int
    num_functions_per_polygon: int
    element_type: int
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
    boundary_method_type_g: int
    boundary_method_type_adf: int
    bubble_type: int


def set_flags(network_input_dimension: int,
              method_order: int,
              method_type: int,
              basis_function_type: int,
              num_functions_per_polygon: int,
              element_type: int,
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
              distribution_points_type: int,
              boundary_method_type_g: int,
              boundary_method_type_adf: int,
              bubble_type: int
              ) -> Flags:

    flags: Flags = {'input_dim': network_input_dimension,
                    'output_dim': 1,
                    'method_order': method_order,
                    'method_type': method_type,
                    'basis_function_type': basis_function_type,
                    'num_functions_per_polygon': num_functions_per_polygon,
                    'element_type': element_type,
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
                    'distribution_points_type': distribution_points_type,
                    'boundary_method_type_g': boundary_method_type_g,
                    'boundary_method_type_adf': boundary_method_type_adf,
                    'bubble_type': bubble_type}

    return flags


def write_flags_on_dictionary(flags: Flags) -> None:
    file = open('{}/dictionary.txt'.format(flags['name_storage']), 'w')
    file.write("method_type = {}\n".format(flags['method_type']))
    file.write("basis_function_type = {}\n".format(flags['basis_function_type']))
    file.write("num_functions_per_polygon = {}\n".format(flags['num_functions_per_polygon']))
    file.write("element_type = {}\n".format(flags['element_type']))
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
    file.write("boundary_method_type_g = {}\n".format(flags['boundary_method_type_g']))
    file.write("boundary_method_type_adf = {}\n".format(flags['boundary_method_type_adf']))
    file.write("bubble_type = {}\n".format(flags['bubble_type']))

    file.close()


def load_flags_from_dictionary(name_storage: str, raw: Dict) -> Flags:
    flags: Flags = {'input_dim': int(raw["input_dim"]),
                    'output_dim': int(raw["output_dim"]),
                    'method_order': int(raw["method_order"]),
                    'method_type': int(raw["method_type"]),
                    'basis_function_type': int(raw["basis_function_type"]),
                    'num_functions_per_polygon': int(raw["num_functions_per_polygon"]),
                    'element_type': int(raw["element_type"]),
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
                    'distribution_points_type': int(raw["distribution_points_type"]),
                    'boundary_method_type_g': int(raw["boundary_method_type_g"]),
                    'boundary_method_type_adf': int(raw["boundary_method_type_adf"]),
                    'bubble_type': int(raw["bubble_type"])}

    return flags


class AbstractBPNAVEM(ABC):

    n_polygons: int
    n_local_pts: int
    n_funcs_per_polygon: int
    one_over_n_funcs: tf.Tensor

    def __init__(self, flags: Flags, in_training: bool = False):

        self.flags = flags
        self.in_training = in_training

        self.layers_list = []

        self.var_phi = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                   validate_shape=False, shape=(None, 1), dtype=tf.float64)
        self.var_phi_grad = tf.Variable(tf.convert_to_tensor([[0., 0.]], dtype=tf.float64), trainable=False,
                                        validate_shape=False, shape=(None, 2), dtype=tf.float64)
        self.var_phi_second_derivatives = tf.Variable(tf.convert_to_tensor([[0., 0., 0.]], dtype=tf.float64),
                                                      trainable=False,
                                                      validate_shape=False, shape=(None, 3), dtype=tf.float64)
        self.var_phi_xx_yy = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                         validate_shape=False, shape=(None, 1), dtype=tf.float64)

        self.var_g = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                 validate_shape=False, shape=(None, None), dtype=tf.float64)
        self.var_g_grad = tf.Variable(tf.convert_to_tensor([[0., 0.]], dtype=tf.float64), trainable=False,
                                      validate_shape=False, shape=(None, 2), dtype=tf.float64)
        self.var_g_second_derivatives = tf.Variable(tf.convert_to_tensor([[0., 0., 0.]], dtype=tf.float64),
                                                    trainable=False,
                                                    validate_shape=False, shape=(None, 3), dtype=tf.float64)
        self.var_g_xx_yy = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                       validate_shape=False, shape=(None, 1), dtype=tf.float64)

        self.exact_one = 0
        self.tf_two = tf.convert_to_tensor(2.0, dtype=tf.float64)

    @abstractmethod
    def load_model(self, name_storage: str):
        pass

    @abstractmethod
    def save_model(self):
        pass

    def internal_call(self, inputs: tf.Tensor) -> tf.Tensor:
        u = self.layers_list[0](inputs)
        for layer in self.layers_list[1:-1]:
            u = layer(u) + u
        return self.layers_list[-1](u)

    def get_u(self, inputs: tf.Tensor) -> tf.Tensor:
        u0 = self.internal_call(inputs)
        return u0 * self.var_phi + self.var_g

    def get_u0_phi_g(self, inputs: tf.Tensor) -> tf.Tensor:
        u0 = self.internal_call(inputs)
        return tf.concat([u0, self.var_phi, self.var_g], 1)

    def get_u_and_du(self, inputs: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as t:
            t.watch(inputs)
            u0 = self.internal_call(inputs)
        u0_grad = t.gradient(u0, inputs)[:, :2]

        u = u0 * self.var_phi + self.var_g
        grad_u = u0_grad * self.var_phi + u0 * self.var_phi_grad + self.var_g_grad
        return tf.concat([u, grad_u], 1)

    def get_du(self, inputs: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as t:
            t.watch(inputs)
            u0 = self.internal_call(inputs)
        u0_grad = t.gradient(u0, inputs)[:, :2]

        return u0_grad * self.var_phi + u0 * self.var_phi_grad + self.var_g_grad

    def get_u0_phi_g_and_du0_d_phi_dg(self, inputs: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as t:
            t.watch(inputs)
            u0 = self.internal_call(inputs)
        u0_grad = t.gradient(u0, inputs)[:, :2]

        return tf.concat([u0, self.var_phi, self.var_g, u0_grad, self.var_phi_grad, self.var_g_grad], 1)

    def get_second_derivatives_u(self, inputs: tf.Tensor) -> tf.Tensor:

        with tf.GradientTape(persistent=False) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                u0 = self.internal_call(inputs)
            grad = tape1.gradient(u0, inputs, output_gradients=tf.ones_like(u0))
        second_derivatives = tape2.batch_jacobian(grad, inputs)

        u0_xx = second_derivatives[:, 0, 0:1]
        u0_xy = second_derivatives[:, 0, 1:2]
        u0_yy = second_derivatives[:, 1, 1:2]

        u_xx = (u0_xx * self.var_phi + self.tf_two * (grad[:, 0:1] * self.var_phi_grad[:, 0:1])
                + u0 * self.var_phi_second_derivatives[:, 0:1]
                + self.var_g_second_derivatives[:, 0:1])

        u_xy = (u0_xy * self.var_phi + grad[:, 0:1] * self.var_phi_grad[:, 1:2]
                + grad[:, 1:2] * self.var_phi_grad[:, 0:1]
                + u0 * self.var_phi_second_derivatives[:, 1:2]
                + self.var_g_second_derivatives[:, 1:2])

        u_yy = (u0_yy * self.var_phi + self.tf_two * (grad[:, 1:2] * self.var_phi_grad[:, 1:2])
                + u0 * self.var_phi_second_derivatives[:, 2:3]
                + self.var_g_second_derivatives[:, 2:3])

        return tf.concat([u_xx, u_xy, u_yy], 1)

    def setup_model_global_input(self,
                                 xy_per_pol: NDArray[np.float64],
                                 vertices: NDArray[np.float64],
                                 jac_per_pol: NDArray[np.float64],
                                 setup_n_derivatives: SetupDerivatives,
                                 geometry_utilities: gedim.GeometryUtilities):


        num_of_functions = self.flags["num_vertices"] - self.exact_one

        eb = EnforcingBoundary(geometry_utilities,
                               method_order=self.flags["method_order"],
                               boundary_method_type_adfs=BoundaryMethodType(self.flags["boundary_method_type_adf"]),
                               boundary_method_type_g=BoundaryMethodType(self.flags["boundary_method_type_g"]),
                               bubble_type=BubbleType(self.flags["bubble_type"]))

        vertices = tf.transpose(tf.convert_to_tensor(vertices, dtype=tf.float64), perm=[0, 2, 1])
        xy_per_pol = tf.convert_to_tensor(xy_per_pol, dtype=tf.float64)
        jac_per_pol = tf.convert_to_tensor(jac_per_pol, dtype=tf.float64)

        eb.initialize_boundary_properties(vertices)

        g = tf.constant(0, dtype=tf.float64)
        phi = None
        phi_grad = None
        g_grad = tf.constant(0, dtype=tf.float64)
        phi_sec_ders = None
        g_sec_ders = tf.constant(0, dtype=tf.float64)

        match setup_n_derivatives:
            case SetupDerivatives.basis:
                phi = eb.compute_adfs(xy_per_pol)
                g = eb.compute_g(xy_per_pol)

            case SetupDerivatives.basis_and_derivatives:

                phi, phi_grad = eb.compute_adfs_and_grads(xy_per_pol)
                phi_grad = eb.map_adf_grads(phi_grad, jac_per_pol)

                g, g_grad = eb.compute_g_and_grads(xy_per_pol)
                g_grad = eb.map_g_grads(g_grad, jac_per_pol)

            case SetupDerivatives.basis_and_derivatives_and_laplacian:

                g, g_grad, g_sec_ders = eb.compute_g_and_grads_and_second_derivatives(xy_per_pol)
                g_grad = eb.map_g_grads(g_grad, jac_per_pol)
                g_sec_ders = eb.map_g_second_derivatives(g_sec_ders, jac_per_pol)

                phi, phi_grad, phi_sec_ders = eb.compute_adfs_and_grads_and_second_derivatives(xy_per_pol)
                phi_grad = eb.map_adf_grads(phi_grad, jac_per_pol)
                phi_sec_ders = eb.map_adf_second_derivatives(phi_sec_ders, jac_per_pol)

            case _:
                raise ValueError("not valid setup_n_derivatives.")

        g = g[:, :, :num_of_functions]

        self.n_polygons, self.n_local_pts, self.n_funcs_per_polygon = g.shape
        self.one_over_n_funcs = tf.convert_to_tensor(1.0 / (self.n_polygons * self.n_funcs_per_polygon), dtype=tf.float64)

        phi = tf.tile(phi, [1, 1, self.n_funcs_per_polygon])
        phi = tf.transpose(phi, [0, 2, 1])
        phi = tf.reshape(phi, [-1, 1])
        self.var_phi.assign(phi)

        g = tf.transpose(g, [0, 2, 1])
        g = tf.reshape(g, [-1, 1])
        self.var_g.assign(g)

        match setup_n_derivatives:
            case SetupDerivatives.basis_and_derivatives | SetupDerivatives.basis_and_derivatives_and_laplacian:
                phi_grad = tf.transpose(phi_grad, [0, 2, 1, 3])
                phi_grad = tf.reshape(phi_grad, [-1, 2])
                g_grad = tf.transpose(g_grad, [0, 2, 1, 3])
                g_grad = tf.reshape(g_grad, [-1, 2])

                self.var_phi_grad.assign(phi_grad)
                self.var_g_grad.assign(g_grad)
            case SetupDerivatives.basis:
                pass
            case _:
                raise ValueError("not valid setup_n_derivatives.")

        match setup_n_derivatives:
            case SetupDerivatives.basis_and_derivatives_and_laplacian:
                phi_sec_ders = tf.transpose(phi_sec_ders, [0, 2, 1, 3])
                self.var_phi_second_derivatives = tf.reshape(phi_sec_ders, [-1, 3])
                g_sec_ders = tf.transpose(g_sec_ders, [0, 2, 1, 3])
                self.var_g_second_derivatives = tf.reshape(g_sec_ders, [-1, 3])

                self.var_phi_xx_yy.assign(
                    self.var_phi_second_derivatives[:, 0:1] + self.var_phi_second_derivatives[:, 2:3])
                self.var_g_xx_yy.assign(self.var_g_second_derivatives[:, 0:1] + self.var_g_second_derivatives[:, 2:3])
            case SetupDerivatives.basis | SetupDerivatives.basis_and_derivatives:
                pass
            case _:
                raise ValueError("not valid setup_n_derivatives.")
