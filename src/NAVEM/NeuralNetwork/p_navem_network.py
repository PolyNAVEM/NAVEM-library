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

import numpy as np
from numpy.typing import NDArray
from NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import Flags, AbstractBPNAVEM

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.layers import Dense


class PNAVEMNetwork(tf.keras.Model, AbstractBPNAVEM):

    def __init__(self, flags: Flags, in_training: bool = False):

        tf.keras.Model.__init__(self, name="p_navem_neural_network")
        AbstractBPNAVEM.__init__(self, flags, in_training)

        self.reg_coefficient = self.flags['regularization_coefficient']

        self.layers_list = [Dense(self.flags['num_neurons_per_layer'],
                                  activation='tanh',
                                  kernel_initializer='glorot_normal')]
        for _ in range(self.flags['num_hidden_layers']):
            self.layers_list.append(Dense(self.flags['num_neurons_per_layer'],
                                          activation='tanh',
                                          kernel_initializer='glorot_normal'))
        self.layers_list.append(Dense(self.flags['output_dim'], kernel_initializer='glorot_normal'))

        self.best_loss = tf.Variable(tf.convert_to_tensor(np.inf, dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.best_w = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False,
                                  shape=(None,), dtype=tf.float64)
        self.bfgs_regularization_grads = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False,
                                                     validate_shape=False, shape=(None,),
                                                     dtype=tf.float64)

        self.loss_one = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                    validate_shape=False, dtype=tf.float64)
        self.loss_x = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                  validate_shape=False, dtype=tf.float64)
        self.loss_y = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                  validate_shape=False, dtype=tf.float64)
        self.loss_one_dx = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                       validate_shape=False, dtype=tf.float64)
        self.loss_x_dx = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                     validate_shape=False, dtype=tf.float64)
        self.loss_y_dx = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                     validate_shape=False, dtype=tf.float64)
        self.loss_one_dy = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                       validate_shape=False, dtype=tf.float64)
        self.loss_x_dy = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                     validate_shape=False, dtype=tf.float64)
        self.loss_y_dy = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                     validate_shape=False, dtype=tf.float64)
        self.training_quad_w = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, 1), dtype=tf.float64)

        self.tf_one = tf.convert_to_tensor(1.0, dtype=tf.float64)

        self.exact_one = int(self.in_training)

        self.x_vertices = None
        self.y_vertices = None
        self.jac_inverse_00 = None
        self.jac_inverse_01 = None
        self.jac_inverse_10 = None
        self.jac_inverse_11 = None
        self.target_x_vals = None
        self.target_y_vals = None
        self.x_lin_coefficients = None
        self.y_lin_coefficients = None

    def build(self, input_shape: tuple[None, int]):
        self.layers_list[0].build(input_shape)
        for layer in self.layers_list[1:]:
            layer.build((None, self.flags["num_neurons_per_layer"]))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.flags["copy_basis_in_train"]:
            return self.get_u_and_du(inputs)
        else:
            return self.get_du(inputs)

    def store_terms_for_loss(self, xy_per_pol: NDArray[np.float64],
                             vertices_per_pol: NDArray[np.float64],
                             jac_inv_per_pol: NDArray[np.float64]):

        vertices_per_pol = tf.convert_to_tensor(vertices_per_pol, dtype=tf.float64)
        x_vertices = tf.expand_dims(vertices_per_pol[:, 0, :], 2)[:, :, :, None, None]
        y_vertices = tf.expand_dims(vertices_per_pol[:, 1, :], 2)[:, :, :, None, None]

        x_n = x_vertices[:, -1:, :, :, :]
        y_n = y_vertices[:, -1:, :, :, :]

        xy_per_pol = tf.convert_to_tensor(xy_per_pol, dtype=tf.float64)
        self.target_x_vals = tf.expand_dims(xy_per_pol[:, :, 0], 1)[:, :, :, None, None] - x_n
        self.target_y_vals = tf.expand_dims(xy_per_pol[:, :, 1], 1)[:, :, :, None, None] - y_n
        self.x_lin_coefficients = x_vertices[:, :-1, :, :, :] - x_n
        self.y_lin_coefficients = y_vertices[:, :-1, :, :, :] - y_n

        jac_inverse = tf.convert_to_tensor(jac_inv_per_pol, dtype=tf.float64)
        self.jac_inverse_00 = jac_inverse[:, 0, 0, :, None, None, None]
        self.jac_inverse_01 = jac_inverse[:, 0, 1, :, None, None, None]
        self.jac_inverse_10 = jac_inverse[:, 1, 0, :, None, None, None]
        self.jac_inverse_11 = jac_inverse[:, 1, 1, :, None, None, None]

    # @tf.function
    def internal_pol_loss(self, _y_true: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
        y_predicted_reshaped = tf.reshape(y_predicted, [self.n_polygons, self.n_funcs_per_polygon, self.n_local_pts, 1, 1])
        return self.copy_pols(y_predicted_reshaped)

    def internal_pol_and_grad_loss(self, _y_true: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
        y_predicted_reshaped = tf.reshape(y_predicted, [self.n_polygons, self.n_funcs_per_polygon, self.n_local_pts, 1, 3])
        pol_loss = self.copy_pols(y_predicted_reshaped[:, :, :, :, :1])
        grad_loss = self.copy_grads(y_predicted_reshaped[:, :, :, :, 1:])
        return pol_loss + grad_loss

    def internal_grad_loss(self, _y_true: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
        y_predicted_reshaped = tf.reshape(y_predicted, [self.n_polygons, self.n_funcs_per_polygon, self.n_local_pts, 1, 2])
        return self.copy_grads(y_predicted_reshaped)

    def copy_pols(self, bases: tf.Tensor) -> tf.Tensor:
        comb_for_x = tf.reduce_sum(bases * self.x_lin_coefficients, axis=1, keepdims=True)
        comb_for_y = tf.reduce_sum(bases * self.y_lin_coefficients, axis=1, keepdims=True)

        diff_for_x = tf.reduce_mean(tf.square(comb_for_x - self.target_x_vals))
        diff_for_y = tf.reduce_mean(tf.square(comb_for_y - self.target_y_vals))

        self.loss_x.assign(diff_for_x)
        self.loss_y.assign(diff_for_y)
        return diff_for_x + diff_for_y

    def copy_grads(self, rot_grads: tf.Tensor) -> tf.Tensor:
        rot_dx = rot_grads[:, :, :, :, :1]
        rot_dy = rot_grads[:, :, :, :, 1:]

        bases_dx = self.jac_inverse_00 * rot_dx + self.jac_inverse_10 * rot_dy
        bases_dy = self.jac_inverse_01 * rot_dx + self.jac_inverse_11 * rot_dy

        comb_for_x_dx = tf.reduce_sum(bases_dx * self.x_lin_coefficients, axis=1, keepdims=True)
        comb_for_y_dx = tf.reduce_sum(bases_dx * self.y_lin_coefficients, axis=1, keepdims=True)

        comb_for_x_dy = tf.reduce_sum(bases_dy * self.x_lin_coefficients, axis=1, keepdims=True)
        comb_for_y_dy = tf.reduce_sum(bases_dy * self.y_lin_coefficients, axis=1, keepdims=True)

        diff_for_x_dx = tf.reduce_mean(tf.square(comb_for_x_dx - self.tf_one))
        diff_for_y_dx = tf.reduce_mean(tf.square(comb_for_y_dx))

        diff_for_x_dy = tf.reduce_mean(tf.square(comb_for_x_dy))
        diff_for_y_dy = tf.reduce_mean(tf.square(comb_for_y_dy - self.tf_one))

        self.loss_x_dx.assign(diff_for_x_dx)
        self.loss_y_dx.assign(diff_for_y_dx)
        self.loss_x_dy.assign(diff_for_x_dy)
        self.loss_y_dy.assign(diff_for_y_dy)

        return diff_for_x_dx + diff_for_y_dx + diff_for_x_dy + diff_for_y_dy

    def save_model(self):
        name_storage = self.flags['name_storage']

        self.flags = None

        zero_1d = tf.convert_to_tensor([0.], dtype=tf.float64)
        zero_2d = tf.convert_to_tensor([[0.]], dtype=tf.float64)
        zero_zero_2d = tf.convert_to_tensor([[0., 0.]], dtype=tf.float64)

        self.best_w.assign(zero_1d)
        self.bfgs_regularization_grads.assign(zero_1d)
        self.training_quad_w.assign(zero_2d)
        self.var_g.assign(zero_2d)
        self.var_phi.assign(zero_2d)
        self.var_g_grad.assign(zero_zero_2d)
        self.var_phi_grad.assign(zero_zero_2d)

        self.x_vertices = None
        self.y_vertices = None
        self.jac_inverse_00 = None
        self.jac_inverse_01 = None
        self.jac_inverse_10 = None
        self.jac_inverse_11 = None
        self.target_x_vals = None
        self.target_y_vals = None
        self.x_lin_coefficients = None
        self.y_lin_coefficients = None

        self.save_weights(name_storage + "/nn_weights.weights.h5")

    def load_model(self, name_storage: str):
        self.build(input_shape=(None, self.flags["input_dim"]))
        self.load_weights(name_storage + "/nn_weights.weights.h5")
