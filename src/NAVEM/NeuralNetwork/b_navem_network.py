import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from src.NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import Flags, AbstractBPNAVEM


class BNAVEMNetwork(tf.keras.Model, AbstractBPNAVEM):

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
                     self.tf_two * (grad[:, 0:1] * self.var_phi_grad[:, 0:1] + grad[:, 1:2] * self.var_phi_grad[:, 1:2]) +
                     u0 * self.var_phi_xx_yy + self.var_g_xx_yy)
        return laplacian

    def get_second_derivatives_u(self, inputs):

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

    def get_u_and_du(self, inputs):
        with tf.GradientTape() as t2:
            t2.watch(inputs)
            u0 = self.internal_call(inputs)
        grad = t2.gradient(u0, inputs)

        u = u0 * self.var_phi + self.var_g
        u_x = grad[:, 0:1] * self.var_phi + u0 * self.var_phi_grad[:, 0:1] + self.var_g_grad[:, 0:1]
        u_y = grad[:, 1:2] * self.var_phi + u0 * self.var_phi_grad[:, 1:2] + self.var_g_grad[:, 1:2]
        return tf.concat([u, u_x, u_y], 1)

    def get_u0_phi_g_and_du0_d_phi_dg(self, inputs):
        with tf.GradientTape() as t2:
            t2.watch(inputs)
            u0 = self.internal_call(inputs)
        grad = t2.gradient(u0, inputs)

        return tf.concat([u0, self.var_phi, self.var_g, grad[:, 0:1], grad[:, 1:2],
                          self.var_phi_grad[:, 0:1], self.var_phi_grad[:, 1:2], self.var_g_grad[:, 0:1], self.var_g_grad[:, 1:2]], 1)

    def get_u(self, inputs):
        u0 = self.internal_call(inputs)
        return u0 * self.var_phi + self.var_g

    def get_u0_phi_g(self, inputs):
        u0 = self.internal_call(inputs)
        return tf.concat([u0, self.var_phi, self.var_g], 1)

    # @tf.function
    def pinn_laplace_loss(self, _y_true, y_predicted):
        integral_argument = tf.square(y_predicted)
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

        self.save_weights(name_storage + "/nn_weights.weights.h5")


    def load_model(self, name_storage: str):
        self.build(input_shape=(None, self.flags["input_dim"]))
        self.load_weights(name_storage + "/nn_weights.weights.h5")