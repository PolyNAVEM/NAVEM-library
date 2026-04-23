import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


class NAPEMNetworkGrad(tf.keras.Model):
    def __init__(self, n_verts, input_dim, n_neurons, n_hidden_layers, reg_coeff):
        super(NAPEMNetworkGrad, self).__init__(name='napem_network_gradient')

        self.input_dim = input_dim
        self.n_verts = n_verts
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.reg_coefficient = reg_coeff

        self.layers_list_dx = [Dense(self.n_neurons, input_dim=self.input_dim, activation='tanh', kernel_initializer='glorot_normal')]
        for _ in range(self.n_hidden_layers):
            self.layers_list_dx.append(Dense(self.n_neurons, input_dim=self.n_neurons, activation='tanh', kernel_initializer='glorot_normal'))
        self.layers_list_dx.append(Dense(1, input_dim=self.n_neurons, kernel_initializer='glorot_normal'))

        self.layers_list_dy = [Dense(self.n_neurons, input_dim=self.input_dim, activation='tanh', kernel_initializer='glorot_normal')]
        for _ in range(self.n_hidden_layers):
            self.layers_list_dy.append(Dense(self.n_neurons, input_dim=self.n_neurons, activation='tanh', kernel_initializer='glorot_normal'))
        self.layers_list_dy.append(Dense(1, input_dim=self.n_neurons, kernel_initializer='glorot_normal'))

        self.best_loss = tf.Variable(tf.convert_to_tensor(np.Inf, dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.best_w = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False, shape=(None,), dtype=tf.float64)
        self.bfgs_regularization_grads = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False, shape=(None,),
                                                     dtype=tf.float64)

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
        self.loss_copy = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                            validate_shape=False, dtype=tf.float64)

        self.output_scaling = tf.Variable(tf.convert_to_tensor([0., 0.], dtype=tf.float64), trainable=False,
                                            validate_shape=False, dtype=tf.float64)

        self.tf_one = tf.convert_to_tensor(1.0, dtype=tf.float64)

    @tf.function
    def call(self, inputs):
        return self.get_du(inputs)

    def get_du(self, inputs):
        du_dx = self.layers_list_dx[0](inputs)
        for layer in self.layers_list_dx[1:-1]:
            du_dx = layer(du_dx) + du_dx
        du_dx = self.layers_list_dx[-1](du_dx)

        du_dy = self.layers_list_dy[0](inputs)
        for layer in self.layers_list_dy[1:-1]:
            du_dy = layer(du_dy) + du_dy
        du_dy = self.layers_list_dy[-1](du_dy)
        return tf.concat([du_dx, du_dy], axis=1) * self.output_scaling

    def store_terms_for_loss(self, func_pred_grad, vertices_per_pol, jac_inv_per_pol):
        self.func_pred_grad = func_pred_grad
        self.scaling_func_pred_grad = 1.0 / tf.reduce_mean(tf.square(func_pred_grad))
        self.output_scaling.assign(tf.reduce_mean(tf.abs(func_pred_grad), axis=0))
        self.x_verts = tf.expand_dims(vertices_per_pol[:, 0, :], 2)
        self.y_verts = tf.expand_dims(vertices_per_pol[:, 1, :], 2)
        self.jac_invs = tf.convert_to_tensor(jac_inv_per_pol, dtype=tf.float64)

    def inter_grad_loss(self, y_true, y_pred):
        rot_bases_dx = tf.reshape(y_pred[:, 0], [self.n_pols, self.n_verts, self.n_local_pts])
        rot_bases_dy = tf.reshape(y_pred[:, 1], [self.n_pols, self.n_verts, self.n_local_pts])

        copy_func_grad = tf.reduce_mean(tf.square(y_pred - self.func_pred_grad)) * self.scaling_func_pred_grad

        bases_dx = self.jac_invs[:, 0, 0, :, None] * rot_bases_dx + self.jac_invs[:, 1, 0, :, None] * rot_bases_dy
        bases_dy = self.jac_invs[:, 0, 1, :, None] * rot_bases_dx + self.jac_invs[:, 1, 1, :, None] * rot_bases_dy

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
        self.loss_copy.assign(copy_func_grad)

        return (diff_for_one_dx + diff_for_x_dx + diff_for_y_dx +
                diff_for_one_dy + diff_for_x_dy + diff_for_y_dy + 1e-1 * copy_func_grad)

        # return copy_func_grad
