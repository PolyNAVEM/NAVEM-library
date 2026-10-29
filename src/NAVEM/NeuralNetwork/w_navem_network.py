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
from NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import (Flags, AbstractBPNAVEM)
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.layers import Dense


class WNAVEMNetwork(tf.keras.Model, AbstractBPNAVEM):

    def __init__(self, flags: Flags, in_training: bool = False):

        tf.keras.Model.__init__(self, name="b_navem_neural_network")
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

        self.curr_laplacian = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                          validate_shape=False, dtype=tf.float64)
        self.training_quad_w = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, 1), dtype=tf.float64)

    def build(self, input_shape: tuple[None, int]):
        self.layers_list[0].build(input_shape)
        for layer in self.layers_list[1:]:
            layer.build((None, self.flags["num_neurons_per_layer"]))

    def wachspress_weights(self,
                           points: tf.Tensor, vertices: tf.Tensor) -> tf.Tensor:
        """
        Compute Wachspress weights as:

            w_i(x) =
                det(n_{i-1}, n_i)
                PROD_{j != i-1,i} h_j(x)

        Parameters
        ----------
        points : (num_points, 2)
        vertices : (num_vertices, 2)

        Returns
        -------
        lambda_w : (num_points, num_vertices)
            normalized Wachspress coordinates
        """

        num_vertices = vertices.shape[1]
        num_points = points.shape[0]

        v_next = tf.roll(vertices, shift=-1, axis=1)

        # edge vectors
        edge_tangents = v_next - vertices

        # NON normalized normals
        edge_normals = tf.stack([-edge_tangents[:, :, 1], edge_tangents[:, :, 0]], axis=2)

        # h_i(x)
        h = tf.reduce_sum(
            (points[:, None, :] - vertices[:, :, :]) *
            edge_normals[:, :, :],
            axis=-1
        )

        prev_edge_normals = tf.roll(edge_normals, shift=1, axis=1)
        det_prev_current = tf.einsum('pij,pij->pi',
                                     prev_edge_normals[:, :, ::-1],
                                     edge_normals * tf.constant([1.0, -1.0], dtype=tf.float64))

        prod_dist = tf.zeros(shape=(num_points, 0), dtype=tf.float64)
        for i in range(num_vertices):
            mask = [j for j in range(num_vertices) if j != i and j != (i - 1) % num_vertices]

            prod_im1_i = tf.expand_dims(tf.reduce_prod(tf.gather(h, mask, axis=1), axis=1), axis=1)

            prod_dist = tf.concat([prod_dist, prod_im1_i], axis=1)

        w = prod_dist * det_prev_current[:, :]

        return w / tf.reduce_sum(w, axis=1, keepdims=True)

    def compute_network_lifting(self,
                                inputs: tf.Tensor,
                                boundary_function) -> tf.Tensor:


        num_vertices = self.flags["num_vertices"]

        # Index vectors holding the value of i, ip1, im1 at each loop iteration
        i = tf.range(num_vertices)
        ip1 = (i + 1) % num_vertices
        im1 = (i - 1) % num_vertices

        first = tf.expand_dims(tf.tile(tf.constant([[1.0, 0.0]], dtype=inputs.dtype), [tf.shape(inputs)[0], 1]), axis=1)
        vertices = tf.transpose(tf.reshape(inputs[:, 2:], (tf.shape(inputs)[0], 2, num_vertices - 1)), perm=[0, 2, 1])
        vertices = tf.concat([first, vertices], axis=1)


        x_i = vertices[:, :, 0]
        y_i = vertices[:, :, 1]
        x_ip1_v = tf.gather(vertices, ip1, axis=1)[:, :, 0]
        y_ip1_v = tf.gather(vertices, ip1, axis=1)[:, :, 1]
        x_im1_v = tf.gather(vertices, im1, axis=1)[:, :, 0]
        y_im1_v = tf.gather(vertices, im1, axis=1)[:, :, 1]

        points = inputs[:, :2]
        lambdas = self.wachspress_weights(points, vertices)

        # Lambdas associated with the next / previous vertices (size [polygons, points, vertices])
        lam_ip1 = tf.gather(lambdas, ip1, axis=1)
        lam_im1 = tf.gather(lambdas, im1, axis=1)

        # (x,y) coordinates where the boundary function will be evaluated
        x_ip1 = (1.0 - lam_ip1) * x_i + lam_ip1 * x_ip1_v
        y_ip1 = (1.0 - lam_ip1) * y_i + lam_ip1 * y_ip1_v

        x_im1 = (1.0 - lam_im1) * x_i + lam_im1 * x_im1_v
        y_im1 = (1.0 - lam_im1) * y_i + lam_im1 * y_im1_v

        # Contributions from every vertex at once
        term_a = boundary_function(i, x_ip1, y_ip1)
        term_b = boundary_function(im1, x_im1, y_im1)
        term_c = boundary_function(i, vertices[:, :, 0], vertices[:, :, 1])

        # Sum over vertices
        g = tf.reduce_sum(lambdas * (term_a + term_b - term_c), axis=1)

        return g


    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        e_x = tf.concat([tf.ones_like(inputs[:, 0:1]), tf.zeros_like(inputs[:, 1:])], axis=1)
        e_y = tf.concat([tf.zeros_like(inputs[:, 0:1]), tf.ones_like(inputs[:, 0:1]),
                         tf.zeros_like(inputs[:, 2:])], axis=1)

        with tf.autodiff.ForwardAccumulator(inputs, e_x) as acc_x:
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                u0 = self.internal_call(inputs)
            grad = tape1.gradient(u0, inputs)
        u_xx = acc_x.jvp(grad)[:, 0:1]

        with tf.autodiff.ForwardAccumulator(inputs, e_y) as acc_y:
            with tf.GradientTape() as tape1b:
                tape1b.watch(inputs)
                u0b = self.internal_call(inputs)
            grad_b = tape1b.gradient(u0b, inputs)
        u_yy = acc_y.jvp(grad_b)[:, 1:2]

        laplacian_u0 = u_xx + u_yy

        laplacian = (laplacian_u0 * self.var_phi +
                     self.tf_two * (
                             grad[:, 0:1] * self.var_phi_grad[:, 0:1] +
                             grad[:, 1:2] * self.var_phi_grad[:, 1:2]) +
                     u0 * self.var_phi_xx_yy + self.var_g_xx_yy)

        return laplacian

    # @tf.function
    # def call(self, inputs: tf.Tensor) -> tf.Tensor:
    #
    #     e_x = tf.concat([tf.ones_like(inputs[:, 0:1]), tf.zeros_like(inputs[:, 1:])], axis=1)
    #     e_y = tf.concat([tf.zeros_like(inputs[:, 0:1]), tf.ones_like(inputs[:, 0:1]),
    #                      tf.zeros_like(inputs[:, 2:])], axis=1)
    #
    #     with tf.autodiff.ForwardAccumulator(inputs, e_x) as acc_x:
    #         with tf.autodiff.ForwardAccumulator(inputs, e_y) as acc_y:
    #             with tf.GradientTape() as tape1:
    #                 tape1.watch(inputs)
    #                 u0 = self.internal_call(inputs)
    #             grad = tape1.gradient(u0, inputs)
    #         grad_jvp_y = acc_y.jvp(grad)
    #     grad_jvp_x = acc_x.jvp(grad)
    #
    #     u_xx = grad_jvp_x[:, 0:1]
    #     u_yy = grad_jvp_y[:, 1:2]
    #
    #     laplacian_u0 = u_xx + u_yy
    #
    #     laplacian = (laplacian_u0 * self.var_phi +
    #                  self.tf_two * (
    #                          grad[:, 0:1] * self.var_phi_grad[:, 0:1] +
    #                          grad[:, 1:2] * self.var_phi_grad[:, 1:2]) +
    #                  u0 * self.var_phi_xx_yy + self.var_g_xx_yy)
    #     return laplacian

    def pinn_laplace_loss(self, _y_true: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
        integral_argument = tf.square(y_predicted - self.laplacian_basis_functions)
        integrals = tf.reduce_sum(integral_argument) * self.one_over_n_funcs
        self.curr_laplacian.assign(integrals)
        return integrals

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
        self.var_g_grad.assign(zero_zero_2d)
        self.var_g_xx_yy.assign(zero_2d)
        self.var_phi.assign(zero_2d)
        self.var_phi_grad.assign(zero_zero_2d)
        self.var_phi_xx_yy.assign(zero_2d)
        self.laplacian_basis_functions.assign(zero_2d)

        self.save_weights(name_storage + "/nn_weights.weights.h5")

    def load_model(self, name_storage: str):
        self.build(input_shape=(None, self.flags["input_dim"]))
        self.load_weights(name_storage + "/nn_weights.weights.h5")
