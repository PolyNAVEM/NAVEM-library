import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from src.NAVEM.Utilities.enforcing_boundary_functions import EnforcingBoundary
from src.NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import Flags
from src.NAVEM.NeuralNetwork.abstract_bp_navem import AbstractBPNAVEM


class PNAVEMNetwork(tf.keras.Model, AbstractBPNAVEM):

    def __init__(self, flags: Flags):

        super(PNAVEMNetwork, self).__init__(name='pnavem_neural_network')

        self.flags = flags

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

        self.var_phi = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                   validate_shape=False, shape=(None, 1), dtype=tf.float64)
        self.var_phi_grad = tf.Variable(tf.convert_to_tensor([[0., 0.]], dtype=tf.float64), trainable=False,
                                        validate_shape=False, shape=(None, 2), dtype=tf.float64)

        self.var_g = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                 validate_shape=False, shape=(None, None), dtype=tf.float64)
        self.var_g_grad = tf.Variable(tf.convert_to_tensor([[0., 0.]], dtype=tf.float64), trainable=False,
                                      validate_shape=False, shape=(None, 2), dtype=tf.float64)

        self.tf_one = tf.convert_to_tensor(1.0, dtype=tf.float64)

        self.x_verts = None
        self.y_verts = None
        self.jac_invs_00 = None
        self.jac_invs_01 = None
        self.jac_invs_10 = None
        self.jac_invs_11 = None
        self.target_x_vals = None
        self.target_y_vals = None
        self.x_lin_coeffs = None
        self.y_lin_coeffs = None

    def build(self, input_shape):
        self.layers_list[0].build(input_shape)
        for layer in self.layers_list[1:]:
            layer.build((None, self.flags["num_neurons_per_layer"]))

    # @tf.function
    def internal_call(self, inputs):
        u = self.layers_list[0](inputs)
        for layer in self.layers_list[1:-1]:
            u = layer(u) + u
        return self.layers_list[-1](u)

    # @tf.function
    def call(self, inputs):
        return self.call_for_training(inputs)

    def get_u_and_du(self, inputs):
        with tf.GradientTape() as t:
            t.watch(inputs)
            u0 = self.internal_call(inputs)
        u0_grad = t.gradient(u0, inputs)[:, :2]

        u = u0 * self.var_phi + self.var_g
        grad_u = u0_grad * self.var_phi + u0 * self.var_phi_grad + self.var_g_grad
        return tf.concat([u, grad_u], 1)

    # @tf.function
    def get_du(self, inputs):
        with tf.GradientTape() as t:
            t.watch(inputs)
            u0 = self.internal_call(inputs)
        u0_grad = t.gradient(u0, inputs)[:, :2]

        return u0_grad * self.var_phi + u0 * self.var_phi_grad + self.var_g_grad

    def get_u0_phi_g_and_du0_dphi_dg(self, inputs):
        with tf.GradientTape() as t:
            t.watch(inputs)
            u0 = self.internal_call(inputs)
        u0_grad = t.gradient(u0, inputs)[:, :2]

        return tf.concat([u0, self.var_phi, self.var_g, u0_grad, self.var_phi_grad, self.var_g_grad], 1)

    def get_u(self, inputs):
        u0 = self.internal_call(inputs)
        return u0 * self.var_phi + self.var_g

    def get_u0_phi_g(self, inputs):
        u0 = self.internal_call(inputs)
        return tf.concat([u0, self.var_phi, self.var_g], 1)

    def store_terms_for_loss(self, xy_per_pol, vertices_per_pol, jac_inv_per_pol):
        x_verts = tf.expand_dims(vertices_per_pol[:, 0, :], 2)[:, :, :, None, None]
        y_verts = tf.expand_dims(vertices_per_pol[:, 1, :], 2)[:, :, :, None, None]

        x_N = x_verts[:, -1:, :, :, :]
        y_N = y_verts[:, -1:, :, :, :]

        self.target_x_vals = tf.expand_dims(xy_per_pol[:, :, 0], 1)[:, :, :, None, None] - x_N
        self.target_y_vals = tf.expand_dims(xy_per_pol[:, :, 1], 1)[:, :, :, None, None] - y_N
        self.x_lin_coeffs = x_verts[:, :-1, :, :, :] - x_N
        self.y_lin_coeffs = y_verts[:, :-1, :, :, :] - y_N

        jac_invs = tf.convert_to_tensor(jac_inv_per_pol, dtype=tf.float64)
        self.jac_invs_00 = jac_invs[:, 0, 0, :, None, None, None]
        self.jac_invs_01 = jac_invs[:, 0, 1, :, None, None, None]
        self.jac_invs_10 = jac_invs[:, 1, 0, :, None, None, None]
        self.jac_invs_11 = jac_invs[:, 1, 1, :, None, None, None]

    def setup_model_global_input(self, xy_per_pol, vertices, jac_per_pol, setup_n_derivatives, geometry_utilities):
        eb = EnforcingBoundary(geometry_utilities, method_order=self.flags["method_order"])
        eb.prepare_using_vertices(vertices, jac_per_pol)

        xy_per_pol = tf.convert_to_tensor(xy_per_pol, dtype=tf.float64)
        num_of_functions = self.flags["num_vertices"] - 1

        if setup_n_derivatives == 0:
            phi, g = eb.phi_and_g(xy_per_pol)
        elif setup_n_derivatives == 1:
            phi, g, phi_grad, g_grad = eb.phi_and_g_and_grads(xy_per_pol)
            g_grad = g_grad[:, :, :num_of_functions, :]
            phi_grad, g_grad = eb.map_phi_and_g_grads(phi_grad, g_grad)
        else:
            raise ValueError("setup_n_derivatives must be 0 or 1.")
        g = g[:, :, :num_of_functions]

        self.n_pols, self.n_local_pts, self.n_funcs_per_pol = g.shape

        phi = tf.tile(phi, [1, 1, self.n_funcs_per_pol])
        phi = tf.transpose(phi, [0, 2, 1])
        phi = tf.reshape(phi, [-1, 1])
        g = tf.transpose(g, [0, 2, 1])
        g = tf.reshape(g, [-1, 1])

        self.var_phi.assign(phi)
        self.var_g.assign(g)

        if setup_n_derivatives == 1:
            phi_grad = tf.transpose(phi_grad, [0, 2, 1, 3])
            phi_grad = tf.reshape(phi_grad, [-1, 2])
            g_grad = tf.transpose(g_grad, [0, 2, 1, 3])
            g_grad = tf.reshape(g_grad, [-1, 2])

            self.var_phi_grad.assign(phi_grad)
            self.var_g_grad.assign(g_grad)

    # @tf.function
    def internal_pol_loss(self, y_true, y_pred):
        y_pred_reshaped = tf.reshape(y_pred, [self.n_pols, self.n_funcs_per_pol, self.n_local_pts, 1, 1])
        return self.copy_pols(y_pred_reshaped)

    def internal_pol_and_grad_loss(self, y_true, y_pred):
        y_pred_reshaped = tf.reshape(y_pred, [self.n_pols, self.n_funcs_per_pol, self.n_local_pts, 1, 3])
        pol_loss = self.copy_pols(y_pred_reshaped[:, :, :, :, :1])
        grad_loss = self.copy_grads(y_pred_reshaped[:, :, :, :, 1:])
        return pol_loss + grad_loss

    def internal_grad_loss(self, y_true, y_pred):
        y_pred_reshaped = tf.reshape(y_pred, [self.n_pols, self.n_funcs_per_pol, self.n_local_pts, 1, 2])
        return self.copy_grads(y_pred_reshaped)

    def copy_pols(self, bases):
        comb_for_x = tf.reduce_sum(bases * self.x_lin_coeffs, axis=1, keepdims=True)
        comb_for_y = tf.reduce_sum(bases * self.y_lin_coeffs, axis=1, keepdims=True)

        diff_for_x = tf.reduce_mean(tf.square(comb_for_x - self.target_x_vals))
        diff_for_y = tf.reduce_mean(tf.square(comb_for_y - self.target_y_vals))

        self.loss_x.assign(diff_for_x)
        self.loss_y.assign(diff_for_y)
        return diff_for_x + diff_for_y

    def copy_grads(self, rot_grads):
        rot_dx = rot_grads[:, :, :, :, :1]
        rot_dy = rot_grads[:, :, :, :, 1:]

        bases_dx = self.jac_invs_00 * rot_dx + self.jac_invs_10 * rot_dy
        bases_dy = self.jac_invs_01 * rot_dx + self.jac_invs_11 * rot_dy

        comb_for_x_dx = tf.reduce_sum(bases_dx * self.x_lin_coeffs, axis=1, keepdims=True)
        comb_for_y_dx = tf.reduce_sum(bases_dx * self.y_lin_coeffs, axis=1, keepdims=True)

        comb_for_x_dy = tf.reduce_sum(bases_dy * self.x_lin_coeffs, axis=1, keepdims=True)
        comb_for_y_dy = tf.reduce_sum(bases_dy * self.y_lin_coeffs, axis=1, keepdims=True)

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

        self.x_verts = None
        self.y_verts = None
        self.jac_invs_00 = None
        self.jac_invs_01 = None
        self.jac_invs_10 = None
        self.jac_invs_11 = None
        self.target_x_vals = None
        self.target_y_vals = None
        self.x_lin_coeffs = None
        self.y_lin_coeffs = None

        self.save_weights(name_storage + "/nn_weights.weights.h5")

    def load_model(self, name_storage: str):
        self.build(input_shape=(None, self.flags["input_dim"]))
        self.load_weights(name_storage + "/nn_weights.weights.h5")
