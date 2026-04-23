import numpy as np
import tensorflow as tf
from typing import List, TypedDict
from tensorflow.keras.layers import Dense


class Flags(TypedDict):
    input_dim: int
    output_dim: int
    method_order: int
    num_vertices: int
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
    list_id_vertices_rationals: List[int]
    regularization_coefficient: float
    num_points_on_each_edge: int
    name_storage: str

def set_flags(network_input_dimension: int,
              method_order: int,
              num_vertices: int,
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
              list_id_vertices_rationals: List[int],
              regularization_coefficient: float,
              num_points_on_each_edge: int,
              export_training_data_file_path: str) -> Flags:

    flags: Flags = {'input_dim': network_input_dimension,
                    'output_dim': 1,
                    'method_order': method_order,
                    'num_vertices': num_vertices,
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
                    'list_id_vertices_rationals': list_id_vertices_rationals,
                    'regularization_coefficient': regularization_coefficient,
                    'num_points_on_each_edge': num_points_on_each_edge,
                    'name_storage': export_training_data_file_path}

    return flags

class PolynomialNetwork(tf.keras.Model):
    def __init__(self, input_dim, n_outputs, n_neurons, n_hidden_layers, reg_coefficient, use_sqrt):
        super(PolynomialNetwork, self).__init__(name='polynomial_network')
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.n_outputs = n_outputs
        self.reg_coefficient = reg_coefficient
        self.use_sqrt = use_sqrt

        self.layers_list = [Dense(self.n_neurons, input_dim=self.input_dim, activation='tanh', kernel_initializer='glorot_normal')]
        for _ in range(self.n_hidden_layers):
            self.layers_list.append(Dense(self.n_neurons, input_dim=self.n_neurons, activation='tanh', kernel_initializer='glorot_normal'))
        self.layers_list.append(Dense(self.n_outputs, input_dim=self.n_neurons, kernel_initializer='glorot_normal'))

        self.best_loss = tf.Variable(tf.convert_to_tensor(np.Inf, dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.best_w = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False, shape=(None,), dtype=tf.float64)
        self.bfgs_regularization_grads = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False, shape=(None,),
                                                     dtype=tf.float64)

        self.tangent_x = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                     validate_shape=False,
                                     shape=(None,None), dtype=tf.float64)
        self.tangent_y = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                     validate_shape=False,
                                     shape=(None,None), dtype=tf.float64)
        self.deriv_labels = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False,
                                        validate_shape=False,
                                        shape=(None,None), dtype=tf.float64)
        self.vertex_filter = tf.Variable(tf.convert_to_tensor([[0.]], dtype=tf.float64), trainable=False, validate_shape=False,
                                  shape=(None,None), dtype=tf.float64)

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

    # @tf.function
    def apply_vandermonde(self, x, vander):
        return tf.einsum('pi,pji->pj', x, vander)

    # @tf.function
    def call(self, x):
        x = self.layers_list[0](x)
        for layer in self.layers_list[1:-1]:
            x = layer(x) + x
        return self.layers_list[-1](x)

    # @tf.function
    def copy_value_and_grad(self, y_true, y_pred):
        value_cost = self.copy_value(y_true, y_pred)
        deriv_cost = self.copy_grad(y_true, y_pred)

        return value_cost + tf.square(deriv_cost)

    # @tf.function
    def copy_value(self, y_true, y_pred):
        u = self.apply_vandermonde(y_pred, self.train_vander)
        value_cost = tf.reduce_mean(tf.square(u - tf.squeeze(y_true)))
        self.curr_distance_l2.assign(value_cost)

        # coeff_norm_pol = tf.reduce_mean(tf.abs(y_pred), axis=1)
        # coeff_norm_tot = tf.reduce_mean(tf.square(coeff_norm_pol))
        # value_cost += 1e-4 * coeff_norm_tot
        if self.use_sqrt:
            return tf.sqrt(value_cost)
        else:
            return value_cost

    # @tf.function
    def copy_grad(self, y_true, y_pred):
        ux = self.apply_vandermonde(y_pred, self.train_vander_dx)
        uy = self.apply_vandermonde(y_pred, self.train_vander_dy)
        tangent_derivative = ux * self.tangent_x + uy * self.tangent_y
        deriv_cost = tf.reduce_mean(tf.square(tangent_derivative - self.deriv_labels)*self.vertex_filter)
        self.curr_distance_h1.assign(deriv_cost)

        # coeff_norm_pol = tf.reduce_mean(tf.abs(y_pred), axis=1)
        # coeff_norm_tot = tf.reduce_mean(tf.square(coeff_norm_pol))
        # deriv_cost += 1e-4 * coeff_norm_tot
        if self.use_sqrt:
            return tf.sqrt(deriv_cost)
        else:
            return deriv_cost

    def copy_weights(self, nn):
        fake_input = tf.convert_to_tensor(np.zeros((1, self.input_dim)))
        self.call(fake_input)
        for i in range(len(self.layers_list)):
            self.layers_list[i].set_weights(nn.layers_list[i].get_weights())

class NAVEMNetwork(tf.keras.Model):

    def __init__(self, flags: Flags):
        super(NAVEMNetwork, self).__init__(name='polynomial_plus_harmonic_network')
        self.best_loss = tf.Variable(tf.convert_to_tensor(np.Inf, dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.best_w = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False, shape=(None,), dtype=tf.float64)
        self.bfgs_regularization_grads = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False, validate_shape=False, shape=(None,),
                                                     dtype=tf.float64)

        self.flags = flags

    def build_network(self):

        self.nn_pol = PolynomialNetwork(self.input_dim-2, self.n_tot_polynomials, self.n_pol_neurons,
                                        self.n_pol_hidden_layers, self.net_reg_coef, self.use_sqrt_in_train)
        
        if self.split_deriv:
            self.nn_pol_deriv = PolynomialNetwork(self.input_dim-2, self.n_tot_polynomials, self.n_pol_neurons,
                                                  self.n_pol_hidden_layers, self.net_reg_coef, self.use_sqrt_in_train)

        # if self.input_dim % 2 == 0:
        #     complex_dim = int(0.5 * self.input_dim)
        # else:
        #     complex_dim = int(0.5 * (self.input_dim+1))

        # self.half_dim = int(complex_dim)
        # self.nn_complex = HarmonicNetwork(complex_dim, 1, self.n_harm_neurons, self.n_harm_hidden_layers, "exp")
        # self.nn_complex.doZeros = self.doSetZerosWeightsHarmonic
        # self.nn_complex.create_layers()
        
        #self.nn_complex = HarmonicNetwork_v2(self.input_dim, self.n_harm_neurons, self.n_harm_neurons, self.n_harm_hidden_layers, "exp")
        #self.nn_complex.doZeros = self.doSetZerosWeightsHarmonic
        #self.nn_complex.create_layers()
    
    # @tf.function
    def call(self, x):
        u_pol = self.nn_pol.call(x[:, 2:]) # call without x and y
        # x_real = x[:,::2],  x_imag = x[:,1::2]
        # u_harm = tf.squeeze(self.nn_complex.split_call(x[:,::2], x[:,1::2]))
        # u_harm = self.nn_complex.split_call(x[:, ::2], x[:, 1::2])
        # tf.print(x.shape,tf.shape(self.nn_pol.vandermonde.read_value()), u_pol.shape,u_harm.shape)
        return u_pol# + u_harm
        # return u_pol
        # return u_harm

    # @tf.function
    def predict_coefficients(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x
    
    def define_vandermonde(self, vandermonde):
        self.vandermonde.assign(vandermonde)

    def save_model(self, filename=None, cheap_save=True):
        if cheap_save:

            zero_1d = tf.convert_to_tensor([0.], dtype=tf.float64)
            zero_2d = tf.convert_to_tensor([[0.]], dtype=tf.float64)
            zero_3d = tf.convert_to_tensor([[[0.]]], dtype=tf.float64)
            
            self.best_w.assign(zero_1d)
            self.bfgs_regularization_grads.assign(zero_1d)
            
            self.nn_pol.best_w.assign(zero_1d)
            self.nn_pol.bfgs_regularization_grads.assign(zero_1d)
            self.nn_pol.tangent_x.assign(zero_2d)
            self.nn_pol.tangent_y.assign(zero_2d)
            self.nn_pol.deriv_labels.assign(zero_2d)
            self.nn_pol.vertex_filter.assign(zero_2d+1)
            self.nn_pol.train_vander.assign(zero_3d)
            self.nn_pol.train_vander_dx.assign(zero_3d)
            self.nn_pol.train_vander_dy.assign(zero_3d)
            
            if self.split_deriv:
                self.nn_pol_deriv.best_w.assign(zero_1d)
                self.nn_pol_deriv.bfgs_regularization_grads.assign(zero_1d)
                self.nn_pol_deriv.tangent_x.assign(zero_2d)
                self.nn_pol_deriv.tangent_y.assign(zero_2d)
                self.nn_pol_deriv.deriv_labels.assign(zero_2d)
                self.nn_pol_deriv.vertex_filter.assign(zero_2d+1)
                self.nn_pol_deriv.train_vander.assign(zero_3d)
                self.nn_pol_deriv.train_vander_dx.assign(zero_3d)
                self.nn_pol_deriv.train_vander_dy.assign(zero_3d)
                
        if filename is None:
            tf.print("saving weights on file ", self.flags['name_storage'])
            self.save_weights(self.flags['name_storage'] + ".ckpt")
        else:
            tf.print("saving weights on file ", filename)
            self.save_weights(filename + ".ckpt")
