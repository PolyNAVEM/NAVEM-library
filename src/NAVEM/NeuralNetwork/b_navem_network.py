import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from src.NAVEM.Utilities.enforcing_boundary_functions import EnforcingBoundary
from src.NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import Flags
from src.NAVEM.NeuralNetwork.abstract_bp_navem import AbstractBPNAVEM


class BNAVEMNetwork(tf.keras.Model, AbstractBPNAVEM):

    def __init__(self, flags: Flags):

        super(BNAVEMNetwork, self).__init__(name='b_navem_neural_network')

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

        self.curr_laplacian = tf.Variable(tf.convert_to_tensor(0., dtype=tf.float64), trainable=False,
                                          validate_shape=False, dtype=tf.float64)
        self.training_quad_w = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                           validate_shape=False, shape=(None, 1), dtype=tf.float64)

        self.var_phi = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                   validate_shape=False, shape=(None, 1), dtype=tf.float64)
        self.var_phi_x = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                     validate_shape=False, shape=(None, 1), dtype=tf.float64)
        self.var_phi_y = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                     validate_shape=False, shape=(None, 1), dtype=tf.float64)
        self.var_phi_xx_yy = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                         validate_shape=False, shape=(None, 1), dtype=tf.float64)

        self.var_g = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                 validate_shape=False, shape=(None, None), dtype=tf.float64)
        self.var_g_x = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                   validate_shape=False, shape=(None, 1), dtype=tf.float64)
        self.var_g_y = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                   validate_shape=False, shape=(None, 1), dtype=tf.float64)
        self.var_g_xx_yy = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                       validate_shape=False, shape=(None, 1), dtype=tf.float64)

        self.tf_two = tf.convert_to_tensor(2.0, dtype=tf.float64)

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
        with tf.GradientTape(persistent=False) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                u0 = self.internal_call(inputs)
            grad = tape1.gradient(u0, inputs, output_gradients=tf.ones_like(u0))
        second_derivatives = tape2.batch_jacobian(grad, inputs)
        laplacian_u0 = second_derivatives[:, 0, 0:1] + second_derivatives[:, 1, 1:2]

        # Laplacian computation
        laplacian = (laplacian_u0 * self.var_phi +
                     self.tf_two * (grad[:, 0:1] * self.var_phi_x + grad[:, 1:2] * self.var_phi_y) +
                     u0 * self.var_phi_xx_yy + self.var_g_xx_yy)
        return laplacian

    def get_u_and_du(self, inputs):
        with tf.GradientTape() as t2:
            t2.watch(inputs)
            u0 = self.internal_call(inputs)
        grad = t2.gradient(u0, inputs)

        u = u0 * self.var_phi + self.var_g
        u_x = grad[:, 0:1] * self.var_phi + u0 * self.var_phi_x + self.var_g_x
        u_y = grad[:, 1:2] * self.var_phi + u0 * self.var_phi_y + self.var_g_y
        return tf.concat([u, u_x, u_y], 1)

    def get_u0_phi_g_and_du0_d_phi_dg(self, inputs):
        with tf.GradientTape() as t2:
            t2.watch(inputs)
            u0 = self.internal_call(inputs)
        grad = t2.gradient(u0, inputs)

        return tf.concat([u0, self.var_phi, self.var_g, grad[:, 0:1], grad[:, 1:2],
                          self.var_phi_x, self.var_phi_y, self.var_g_x, self.var_g_y], 1)

    def get_u(self, inputs):
        u0 = self.internal_call(inputs)
        return u0 * self.var_phi + self.var_g

    def get_u0_phi_g(self, inputs):
        u0 = self.internal_call(inputs)
        return tf.concat([u0, self.var_phi, self.var_g], 1)

    def setup_model_global_input(self, xy_per_pol, vertices, jac_per_pol, setup_n_derivatives, geometry_utilities):
        eb = EnforcingBoundary(geometry_utilities, method_order=self.flags["method_order"])
        eb.prepare_using_vertices(vertices, jac_per_pol)

        xy_per_pol = tf.convert_to_tensor(xy_per_pol, dtype=tf.float64)

        if setup_n_derivatives == 0:
            phi, g = eb.phi_and_g(xy_per_pol)
        elif setup_n_derivatives == 1:
            phi, g, phi_grad, g_grad = eb.phi_and_g_and_grads(xy_per_pol)
            phi_grad, g_grad = eb.map_phi_and_g_grads(phi_grad, g_grad)
        elif setup_n_derivatives == 2:
            phi, g, phi_grad, g_grad, phi_sec_ders, g_sec_ders = eb.phi_and_g_and_grads_and_second_derivatives(
                xy_per_pol)
        else:
            raise ValueError("setup_n_derivatives must be 0, 1 or 2.")

        self.n_pols, self.n_local_pts, self.n_verts = g.shape
        self.one_over_n_funcs = tf.convert_to_tensor(1.0 / (self.n_pols * self.n_verts), dtype=tf.float64)

        phi = tf.tile(phi, [1, 1, self.n_verts])
        phi = tf.transpose(phi, [0, 2, 1])
        phi = tf.reshape(phi, [-1, 1])
        self.var_phi.assign(phi)

        if setup_n_derivatives <= 1:
            g = tf.transpose(g, [0, 2, 1])
            g = tf.reshape(g, [-1, 1])
            self.var_g.assign(g)

        if setup_n_derivatives >= 1:
            phi_grad = tf.transpose(phi_grad, [0, 2, 1, 3])
            phi_grad = tf.reshape(phi_grad, [-1, 2])

            self.var_phi_x.assign(phi_grad[:, 0:1])
            self.var_phi_y.assign(phi_grad[:, 1:2])

            if setup_n_derivatives == 1:
                g_grad = tf.transpose(g_grad, [0, 2, 1, 3])
                g_grad = tf.reshape(g_grad, [-1, 2])

                self.var_g_x.assign(g_grad[:, 0:1])
                self.var_g_y.assign(g_grad[:, 1:2])
            else:
                phi_sec_ders = tf.transpose(phi_sec_ders, [0, 2, 1, 3])
                phi_sec_ders = tf.reshape(phi_sec_ders, [-1, 3])
                g_sec_ders = tf.transpose(g_sec_ders, [0, 2, 1, 3])
                g_sec_ders = tf.reshape(g_sec_ders, [-1, 3])

                self.var_phi_xx_yy.assign(phi_sec_ders[:, 0:1] + phi_sec_ders[:, 2:3])
                self.var_g_xx_yy.assign(g_sec_ders[:, 0:1] + g_sec_ders[:, 2:3])

    # @tf.function
    def pinn_laplace_loss(self, y_true, y_pred):
        integral_argument = tf.square(y_pred)
        # integrals = tf.reduce_sum(self.training_quad_w * integral_argument) * self.one_over_n_funcs
        integrals = tf.reduce_sum(integral_argument) * self.one_over_n_funcs
        self.curr_laplacian.assign(integrals)
        return integrals

    def save_model(self):
        name_storage = self.flags['name_storage']

        self.flags = None

        zero_1d = tf.convert_to_tensor([0.], dtype=tf.float64)
        zero_2d = tf.convert_to_tensor([[0.]], dtype=tf.float64)

        self.best_w.assign(zero_1d)
        self.bfgs_regularization_grads.assign(zero_1d)

        self.training_quad_w.assign(zero_2d)
        self.var_g.assign(zero_2d)
        self.var_g_x.assign(zero_2d)
        self.var_g_y.assign(zero_2d)
        self.var_g_xx_yy.assign(zero_2d)
        self.var_phi.assign(zero_2d)
        self.var_phi_x.assign(zero_2d)
        self.var_phi_y.assign(zero_2d)
        self.var_phi_xx_yy.assign(zero_2d)

        self.save_weights(name_storage + "/nn_weights.weights.h5")


    def load_model(self, name_storage: str):
        self.build(input_shape=(None, self.flags["input_dim"]))
        self.load_weights(name_storage + "/nn_weights.weights.h5")