import numpy as np
import tensorflow as tf
from typing import List, TypedDict
from tensorflow.keras.layers import Dense
from pypolydim import gedim, polydim
from src.NAVEM.Utilities.points_generator import reference_points_distribution, PointsSegmentDistributionType, map_pts_from_1d_to_2d

class Flags(TypedDict):
    input_dim: int
    output_dim: int
    method_order: int
    num_vertices: int
    num_generators: int
    mesh_import_path: str
    num_training_polygons: int
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
    list_id_vertices_hanging: List[int]
    regularization_coefficient: float
    num_points_on_each_edge: int
    name_storage: str

def set_flags(network_input_dimension: int,
              method_order: int,
              num_vertices: int,
              num_generators: int,
              mesh_import_path: str,
              num_training_polygons: int,
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
              list_id_vertices_hanging: List[int],
              regularization_coefficient: float,
              num_points_on_each_edge: int,
              export_training_data_file_path: str) -> Flags:

    flags: Flags = {'input_dim': network_input_dimension,
                    'output_dim': 1,
                    'method_order': method_order,
                    'num_vertices': num_vertices,
                    'num_generators': num_generators,
                    'mesh_import_path': mesh_import_path,
                    'num_training_polygons': num_training_polygons,
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
                    'list_id_vertices_hanging': list_id_vertices_hanging,
                    'regularization_coefficient': regularization_coefficient,
                    'num_points_on_each_edge': num_points_on_each_edge,
                    'name_storage': export_training_data_file_path}

    return flags


class BoundaryLoss:

    def __init__(self, geometry_utilities: gedim.GeometryUtilities, n_pts: int, method_order: int, num_vertices: int):

        self.geometry_utilities = geometry_utilities
        self.n_pts = n_pts
        self.method_order = method_order
        self.num_vertices = num_vertices

        # evaluation points
        self.reference_eval_points = reference_points_distribution(0.0, 1.0, self.n_pts, PointsSegmentDistributionType.uniform)

        lobatto_quadrature = gedim.quadrature.Quadrature_GaussLobatto1D()
        quadrature_data = lobatto_quadrature.fill_points_and_weights(self.method_order)
        self.references_do_fs_on_edge = quadrature_data.points[0, :]

        self.lagrange_coefficients = polydim.interpolation.lagrange.lagrange_1_d_coefficients(self.references_do_fs_on_edge)
        lagrange_values = polydim.interpolation.lagrange.lagrange_1_d_values(self.references_do_fs_on_edge,
                                                                             self.lagrange_coefficients,
                                                                             self.reference_eval_points)
        lagrange_derivatives_values = (
            polydim.interpolation.lagrange.lagrange_1_d_derivative_values(self.references_do_fs_on_edge,
                                                                          self.lagrange_coefficients,
                                                                          self.reference_eval_points))


        self.new_pol_evals = np.zeros([n_pts * self.num_vertices, 1])
        self.new_dpol_dx_evals = np.zeros([n_pts * self.num_vertices, 1])
        zero_pol = np.zeros(n_pts)
        # dofs on vertices (evaluations)
        v = 0
        for e in range(self.num_vertices):
            if e == v:  # edge e after vertex v
                self.new_pol_evals[e * n_pts:(e + 1) * n_pts, v] = lagrange_values[:, 0]
                self.new_dpol_dx_evals[e * n_pts:(e + 1) * n_pts, v] = lagrange_derivatives_values[:, 0]
            elif e == self.num_vertices - 1:
                self.new_pol_evals[e * n_pts:(e + 1) * n_pts, v] = lagrange_values[:, -1]
                self.new_dpol_dx_evals[e * n_pts:(e + 1) * n_pts, v] = lagrange_derivatives_values[:, -1]
            else:
                self.new_pol_evals[e * n_pts:(e + 1) * n_pts, v] = zero_pol
                self.new_dpol_dx_evals[e * n_pts:(e + 1) * n_pts, v] = zero_pol
        # dofs on edges (evaluations)
        e = 0  # edge with the support of the basis function
        for e2 in range(self.num_vertices):  # generic edge of the polygon
            for local_p in range(self.method_order - 1):  # number of internal polynomials
                if e == e2:
                    self.new_pol_evals[e2 * n_pts:(e2 + 1) * n_pts, self.num_vertices + e * (self.method_order - 1) + local_p] = \
                    lagrange_values[:, local_p + 1]
                    self.new_dpol_dx_evals[e2 * n_pts:(e2 + 1) * n_pts, self.num_vertices + e * (self.method_order - 1) + local_p] = \
                    lagrange_derivatives_values[:, local_p + 1]
                else:
                    self.new_pol_evals[
                        e2 * n_pts:(e2 + 1) * n_pts, self.num_vertices + e * (self.method_order - 1) + local_p] = zero_pol
                    self.new_dpol_dx_evals[
                        e2 * n_pts:(e2 + 1) * n_pts, self.num_vertices + e * (self.method_order - 1) + local_p] = zero_pol

    def add_0_pols_from_internal_do_fs(self, n_internal_do_fs: int, nodes: List[float]):
        rep_nodes = np.tile(nodes, (n_internal_do_fs, 1))
        zero_pols = np.zeros(self.num_vertices * self.n_pts * n_internal_do_fs)
        return rep_nodes, zero_pols

    def add_polygon(self, vertices: np.ndarray, add_coords: bool = False):
        nodes = np.zeros((2, self.num_vertices * self.n_pts))
        normals = np.zeros((2, self.num_vertices * self.n_pts))
        # mapping the training pts from the reference element to the current one
        for e in range(self.num_vertices):
            v_start = np.expand_dims(vertices[:, e], 1)
            v_end = np.expand_dims(vertices[:, (e + 1) % self.num_vertices], 1)
            # tf.print("vertices",v_start, v_end)
            first_idx, last_idx = [e * self.n_pts, (e + 1) * self.n_pts]
            nodes[:, first_idx:last_idx] = map_pts_from_1d_to_2d(self.reference_eval_points, v_start, v_end)[:2, :]
            normals[:, first_idx:last_idx] = (v_end - v_start)[:2, :]

        nodes = nodes.T
        edge_lenghts = np.sqrt(normals[0, :] ** 2 + normals[1, :] ** 2)
        normals /= edge_lenghts  # normal normalization
        # values of polynomials for boundary dofs
        all_pols = np.squeeze(self.new_pol_evals.T.reshape(-1, 1))
        # values of derivatives for boundary dofs
        all_derivs = np.squeeze(self.new_dpol_dx_evals.T.reshape(-1, 1)) / edge_lenghts

        # tf.print("prev nodes",nodes)
        if add_coords:
            xy_verts = np.expand_dims(vertices[:2, 1:].flatten(), 0)
            rep_xy_verts = np.repeat(xy_verts, nodes.shape[0], axis=0)
            # print("\nnodes\n",nodes)
            # print("\nrep_xy\n",rep_xy_verts)
            nodes = np.concatenate([nodes, rep_xy_verts], axis=1)
        # tf.print("shapes",nodes.shape, all_pols.shape, normals.T.shape, all_derivs.shape)
        # tf.print("nodes",nodes)
        # tf.print("all_pols",all_pols)
        # tf.print("normals.T",normals.T)
        return nodes, all_pols, normals.T, all_derivs

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

        self.best_loss = tf.Variable(tf.convert_to_tensor(np.inf, dtype=tf.float64), trainable=False, dtype=tf.float64)
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
        self.best_loss = tf.Variable(tf.convert_to_tensor(np.inf, dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.best_w = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64), trainable=False,
                                  validate_shape=False, shape=(None,), dtype=tf.float64)
        self.bfgs_regularization_grads = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float64),
                                                     trainable=False, validate_shape=False, shape=(None,),
                                                     dtype=tf.float64)

        self.flags: Flags = flags

        self.nn_basis_function = PolynomialNetwork(self.flags['input_dim'] - 2,
                                                   self.flags['num_generators'],
                                                   self.flags['num_neurons_per_layer'],
                                                   self.flags['num_hidden_layers'],
                                                   self.flags['regularization_coefficient'],
                                                   self.flags['use_sqrt_in_train'])

        self.nn_basis_derivatives = PolynomialNetwork(self.flags['input_dim'] - 2,
                                                      self.flags['num_generators'],
                                                      self.flags['num_neurons_per_layer'],
                                                      self.flags['num_hidden_layers'],
                                                      self.flags['regularization_coefficient'],
                                                      self.flags['use_sqrt_in_train'])

    
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
            
            self.nn_basis_function.best_w.assign(zero_1d)
            self.nn_basis_function.bfgs_regularization_grads.assign(zero_1d)
            self.nn_basis_function.tangent_x.assign(zero_2d)
            self.nn_basis_function.tangent_y.assign(zero_2d)
            self.nn_basis_function.deriv_labels.assign(zero_2d)
            self.nn_basis_function.vertex_filter.assign(zero_2d+1)
            self.nn_basis_function.train_vander.assign(zero_3d)
            self.nn_basis_function.train_vander_dx.assign(zero_3d)
            self.nn_basis_function.train_vander_dy.assign(zero_3d)
            

            self.nn_basis_derivatives.best_w.assign(zero_1d)
            self.nn_basis_derivatives.bfgs_regularization_grads.assign(zero_1d)
            self.nn_basis_derivatives.tangent_x.assign(zero_2d)
            self.nn_basis_derivatives.tangent_y.assign(zero_2d)
            self.nn_basis_derivatives.deriv_labels.assign(zero_2d)
            self.nn_basis_derivatives.vertex_filter.assign(zero_2d+1)
            self.nn_basis_derivatives.train_vander.assign(zero_3d)
            self.nn_basis_derivatives.train_vander_dx.assign(zero_3d)
            self.nn_basis_derivatives.train_vander_dy.assign(zero_3d)
                
        if filename is None:
            tf.print("saving weights on file ", self.flags['name_storage'])
            self.save_weights(self.flags['name_storage'] + ".ckpt")
        else:
            tf.print("saving weights on file ", filename)
            self.save_weights(filename + ".ckpt")
