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
from NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import (Flags, AbstractBPNAVEM, SetupDerivatives)
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.layers import Dense
from pypolydim import gedim
from numpy.typing import NDArray


class WNAVEMNetwork(tf.keras.Model, AbstractBPNAVEM):

    def __init__(self, flags: Flags, in_training: bool = False):

        tf.keras.Model.__init__(self, name="b_navem_neural_network")
        AbstractBPNAVEM.__init__(self, flags, in_training)

        self.mask_drop = None
        self.mask_keep = None
        self.first_vert = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None, None), dtype=tf.float64)
        self.rest_verts = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None, None), dtype=tf.float64)
        self.call_vertices = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None, None), dtype=tf.float64)
        self.v_prev = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None, None), dtype=tf.float64)
        self.v_next = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None, None), dtype=tf.float64)
        self.call_num_points = tf.Variable(tf.convert_to_tensor(0, dtype=tf.int32), trainable=False,
                                          validate_shape=False, dtype=tf.int32)
        self.num_vertices = 0
        self.edge_normals = tf.Variable(tf.convert_to_tensor([[[0.]]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None, None), dtype=tf.float64)
        self.offsets = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None), dtype=tf.float64)
        self.det_prev_current = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, None), dtype=tf.float64)

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

    def wachspress_weights_efficient(self, points):
        h = tf.einsum('pd,pvd->pv', points, self.edge_normals) - self.offsets
        h_masked = h[:, None, :] * self.mask_keep + self.mask_drop
        prod_dist = tf.reduce_prod(h_masked, axis=2)
        w = prod_dist * self.det_prev_current
        return w / tf.reduce_sum(w, axis=1, keepdims=True)

    def u0_minus_lifting(self, points, params):
        lambdas = self.wachspress_weights_efficient(points)  # (P, V)
        lam_next = tf.roll(lambdas, -1, axis=1)
        lam_prev = tf.roll(lambdas, 1, axis=1)

        pt_a = (1 - lam_next[..., None]) * self.call_vertices + lam_next[..., None] * self.v_next
        pt_b = (1 - lam_prev[..., None]) * self.call_vertices + lam_prev[..., None] * self.v_prev
        pt_c = self.call_vertices

        xy = tf.stack([pt_a, pt_b, pt_c], axis=0)
        xy = tf.reshape(tf.transpose(xy, [0, 2, 1, 3]), (-1, 2))
        par = tf.tile(params, [3 * self.num_vertices, 1])

        big = tf.concat([tf.concat([points, params], axis=1),
                         tf.concat([xy, par], axis=1)], axis=0)

        output = self.internal_call(big)  # single network call

        u0 = output[:self.call_num_points, 0]
        ulift = tf.reshape(output[self.call_num_points:, 0], (3, self.num_vertices, self.call_num_points))
        ulift = tf.transpose(ulift, [0, 2, 1])
        term_a, term_b, term_c = ulift[0], ulift[1], ulift[2]

        ln = tf.reduce_sum(lambdas * (term_a + term_b - term_c), axis=1)
        return u0 - ln

    @tf.function
    def call(self, inputs):
        params = inputs[:, 2:]  # constant w.r.t. differentiation
        points = inputs[:, :2]

        with tf.GradientTape(persistent=True) as outer:
            outer.watch(points)
            with tf.GradientTape() as inner:
                inner.watch(points)
                f = self.u0_minus_lifting(points, params)
            grad = inner.gradient(f, points)
            u_x = grad[:, 0:1]
            u_y = grad[:, 1:2]
        u_xx = outer.gradient(u_x, points)[:, 0:1]
        u_yy = outer.gradient(u_y, points)[:, 1:2]

        return u_xx + u_yy + self.var_g_xx_yy

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
        zero_3d = tf.convert_to_tensor([[[0.]]], dtype=tf.float64)
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

        self.mask_drop = None
        self.mask_keep = None
        self.first_vert = None
        self.rest_verts = None
        self.call_vertices.assign(zero_3d)
        self.v_prev.assign(zero_3d)
        self.v_next.assign(zero_3d)
        self.edge_normals.assign(zero_3d)
        self.offsets.assign(zero_2d)
        self.det_prev_current.assign(zero_2d)

        self.save_weights(name_storage + "/nn_weights.weights.h5")

    def load_model(self, name_storage: str):
        self.build(input_shape=(None, self.flags["input_dim"]))
        self.load_weights(name_storage + "/nn_weights.weights.h5")

    def setup_model_global_input_wnavem(self,
                                 xy_per_pol: NDArray[np.float64],
                                 vertices: NDArray[np.float64],
                                 jac_per_pol: NDArray[np.float64],
                                 setup_n_derivatives: SetupDerivatives,
                                 geometry_utilities: gedim.GeometryUtilities,
                                 inputs: tf.Tensor):
        super().setup_model_global_input(xy_per_pol,
                                         vertices,
                                         jac_per_pol,
                                         setup_n_derivatives,
                                         geometry_utilities)
        points = inputs[:, :2]
        params = inputs[:, 2:]
        self.num_vertices = self.flags["num_vertices"]
        self.call_num_points.assign(np.int32(tf.shape(points)[0].numpy()))

        self.first_vert = tf.tile(tf.constant([[[1.0, 0.0]]], dtype=points.dtype), [self.call_num_points, 1, 1])
        self.rest_verts = tf.transpose(tf.reshape(params, (self.call_num_points, 2, self.num_vertices - 1)), [0, 2, 1])
        self.call_vertices.assign(tf.concat([self.first_vert, self.rest_verts], axis=1))

        self.v_next.assign(tf.roll(self.call_vertices, -1, axis=1))
        self.v_prev.assign(tf.roll(self.call_vertices, 1, axis=1))

        # edge vectors and NON normalized normals
        edge_tangents = self.v_next - self.call_vertices
        self.edge_normals.assign(tf.stack([-edge_tangents[:, :, 1], edge_tangents[:, :, 0]], axis=2))

        # v_i . n_i, so that h_i(x) = x . n_i - offsets_i
        self.offsets.assign(tf.reduce_sum(self.call_vertices * self.edge_normals, axis=-1))

        # det(n_{i-1}, n_i) as a scalar product of the (rotated) previous normal with the current one
        prev_edge_normals = tf.roll(self.edge_normals, shift=1, axis=1)
        rot_prev_edge_normals = tf.stack([prev_edge_normals[:, :, 1],
                                          -prev_edge_normals[:, :, 0]], axis=2)
        self.det_prev_current.assign(tf.reduce_sum(rot_prev_edge_normals * self.edge_normals, axis=-1))

        drop = np.zeros((self.num_vertices, self.num_vertices))
        idx = np.arange(self.num_vertices)
        drop[idx, idx] = 1.0
        drop[idx, (idx - 1) % self.num_vertices] = 1.0
        self.mask_drop = tf.constant(drop, tf.float64)
        self.mask_keep = tf.constant(1.0 - drop, tf.float64)

