import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


class NAPEMNetwork(tf.keras.Model):
    def __init__(self, n_verts, input_dim, n_neurons, n_hidden_layers, reg_coeff):
        super(NAPEMNetwork, self).__init__(name='napem_network')

        self.input_dim = input_dim
        self.n_verts = n_verts
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.reg_coefficient = reg_coeff

        self.layers_list = [Dense(self.n_neurons, input_dim=self.input_dim, activation='tanh', kernel_initializer='glorot_normal')]
        for _ in range(self.n_hidden_layers):
            self.layers_list.append(Dense(self.n_neurons, input_dim=self.n_neurons, activation='tanh', kernel_initializer='glorot_normal'))
        self.layers_list.append(Dense(1, input_dim=self.n_neurons, kernel_initializer='glorot_normal'))

        self.best_loss = tf.Variable(tf.convert_to_tensor(np.Inf, dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.best_w = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False, shape=(None,), dtype=tf.float64)
        self.bfgs_regularization_grads = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False, shape=(None,),
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

    # @tf.function
    def internal_call(self, inputs):
        u = self.layers_list[0](inputs)
        for layer in self.layers_list[1:-1]:
            u = layer(u) + u
        return self.layers_list[-1](u)

    @tf.function
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

        # u0 = self.internal_call(inputs)
        # u0_grad = tf.gradients(u0, inputs)[0][:, :2]
        return u0_grad * self.var_phi + u0 * self.var_phi_grad + self.var_g_grad

    def get_u0_phi_g_and_du0_dphi_dg(self, inputs):
        with tf.GradientTape() as t:
            t.watch(inputs)
            u0 = self.internal_call(inputs)
        u0_grad = t.gradient(u0, inputs)[:, :2]

        return tf.concat([u0, self.var_phi, self.var_g, u0_grad,
                          self.var_phi_grad, self.var_g_grad], 1)

    def get_u(self, inputs):
        u0 = self.internal_call(inputs)
        return u0 * self.var_phi + self.var_g

    def get_u0_phi_g(self, inputs):
        u0 = self.internal_call(inputs)
        return tf.concat([u0, self.var_phi, self.var_g], 1)

    def store_terms_for_loss(self, xy_per_pol, vertices_per_pol, jac_inv_per_pol, napem_exact_one):
        if napem_exact_one:
            x_verts = tf.expand_dims(vertices_per_pol[:, 0, :], 2)[:, :, :, None, None]
            y_verts = tf.expand_dims(vertices_per_pol[:, 1, :], 2)[:, :, :, None, None]

            x_N = x_verts[:, -1:, :, :, :]
            y_N = y_verts[:, -1:, :, :, :]

            self.target_x_vals = tf.expand_dims(xy_per_pol[:, :, 0], 1)[:, :, :, None, None] - x_N
            self.target_y_vals = tf.expand_dims(xy_per_pol[:, :, 1], 1)[:, :, :, None, None] - y_N
            self.x_lin_coeffs = x_verts[:, :-1, :, :, :] - x_N
            self.y_lin_coeffs = y_verts[:, :-1, :, :, :] - y_N

        else:
            self.x_verts = tf.expand_dims(vertices_per_pol[:, 0, :], 2)[:, :, :, None, None]
            self.y_verts = tf.expand_dims(vertices_per_pol[:, 1, :], 2)[:, :, :, None, None]

            self.target_x_vals = tf.expand_dims(xy_per_pol[:, :, 0], 1)[:, :, :, None, None]
            self.target_y_vals = tf.expand_dims(xy_per_pol[:, :, 1], 1)[:, :, :, None, None]

        jac_invs = tf.convert_to_tensor(jac_inv_per_pol, dtype=tf.float64)
        self.jac_invs_00 = jac_invs[:, 0, 0, :, None, None, None]
        self.jac_invs_01 = jac_invs[:, 0, 1, :, None, None, None]
        self.jac_invs_10 = jac_invs[:, 1, 0, :, None, None, None]
        self.jac_invs_11 = jac_invs[:, 1, 1, :, None, None, None]


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
        comb_for_one = tf.reduce_sum(bases, axis=1)
        comb_for_x = tf.reduce_sum(bases * self.x_verts, axis=1, keepdims=True)
        comb_for_y = tf.reduce_sum(bases * self.y_verts, axis=1, keepdims=True)

        diff_for_one = tf.reduce_mean(tf.square(comb_for_one - self.tf_one))
        diff_for_x = tf.reduce_mean(tf.square(comb_for_x - self.target_x_vals))
        diff_for_y = tf.reduce_mean(tf.square(comb_for_y - self.target_y_vals))

        self.loss_one.assign(diff_for_one)
        self.loss_x.assign(diff_for_x)
        self.loss_y.assign(diff_for_y)
        return diff_for_one + diff_for_x + diff_for_y


    def copy_pols_exact_one(self, bases):
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

        comb_for_one_dx = tf.reduce_sum(bases_dx, axis=1)
        comb_for_x_dx = tf.reduce_sum(bases_dx * self.x_verts, axis=1, keepdims=True)
        comb_for_y_dx = tf.reduce_sum(bases_dx * self.y_verts, axis=1, keepdims=True)

        comb_for_one_dy = tf.reduce_sum(bases_dy, axis=1)
        comb_for_x_dy = tf.reduce_sum(bases_dy * self.x_verts, axis=1, keepdims=True)
        comb_for_y_dy = tf.reduce_sum(bases_dy * self.y_verts, axis=1, keepdims=True)

        diff_for_one_dx = tf.reduce_mean(tf.square(comb_for_one_dx))
        diff_for_x_dx = tf.reduce_mean(tf.square(comb_for_x_dx - self.tf_one))
        diff_for_y_dx = tf.reduce_mean(tf.square(comb_for_y_dx))

        diff_for_one_dy = tf.reduce_mean(tf.square(comb_for_one_dy))
        diff_for_x_dy = tf.reduce_mean(tf.square(comb_for_x_dy))
        diff_for_y_dy = tf.reduce_mean(tf.square(comb_for_y_dy - self.tf_one))

        self.loss_one_dx.assign(diff_for_one_dx)
        self.loss_x_dx.assign(diff_for_x_dx)
        self.loss_y_dx.assign(diff_for_y_dx)
        self.loss_one_dy.assign(diff_for_one_dy)
        self.loss_x_dy.assign(diff_for_x_dy)
        self.loss_y_dy.assign(diff_for_y_dy)

        return diff_for_one_dx + diff_for_x_dx + diff_for_y_dx + diff_for_one_dy + diff_for_x_dy + diff_for_y_dy


    def copy_grads_exact_one(self, rot_grads):
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
