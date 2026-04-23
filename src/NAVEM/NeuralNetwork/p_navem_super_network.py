import numpy as np
import tensorflow as tf
from src.NAVEM.NeuralNetwork.p_navem_network_basis import NAPEMNetwork
from src.NAVEM.NeuralNetwork.p_navem_network_basis_derivatives import NAPEMNetworkGrad
from src.NAVEM.Utilities.enforcing_boundary_functions import EnforcingBoundary
from src.NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import Flags

class PNAVEMSupernetwork(tf.keras.Model):

    def __init__(self, flags: Flags):
        super(PNAVEMSupernetwork, self).__init__(name='napem_supernetwork')
        self.flags: Flags = flags

        self.n_funcs = self.n_verts - self.napem_exact_one
        self.nn_func = NAPEMNetwork(self.n_verts, self.input_dim, self.n_neurons, self.n_hidden_layers,
                                    self.net_reg_coef)

        if self.split_deriv:
            self.nn_grad = NAPEMNetworkGrad(self.n_verts, self.input_dim, self.n_neurons, self.n_hidden_layers,
                                            self.net_reg_coef)


    @tf.function
    def call(self, inputs):
        return self.nn_func.call(inputs)

    def get_u(self, inputs):
        return self.nn_func.get_u(inputs)

    def get_u_and_du(self, inputs):
        if self.split_deriv:
            u = self.nn_func.get_u(inputs)
            grad_u = self.nn_grad.get_du(inputs)
            return tf.concat([u, grad_u], axis=1)
        else:
            return self.nn_func.get_u_and_du(inputs)

    def setup_model_global_input(self, xy_per_pol, vertices, jac_per_pol, setup_n_derivatives, geometry_utilities):
        eb = EnforcingBoundary(geometry_utilities)
        eb.prepare_using_vertices(vertices, jac_per_pol)

        xy_per_pol = tf.convert_to_tensor(xy_per_pol, dtype=tf.float64)

        if setup_n_derivatives == 0:
            phi, g = eb.phi_and_g(xy_per_pol)
        elif setup_n_derivatives == 1:
            phi, g, phi_grad, g_grad = eb.phi_and_g_and_grads(xy_per_pol)
            g_grad = g_grad[:, :, :self.n_funcs, :]
            phi_grad, g_grad = eb.map_phi_and_g_grads(phi_grad, g_grad)
        else:
            raise ValueError("setup_n_derivatives must be 0 or 1.")
        g = g[:, :, :self.n_funcs]

        self.nn_func.n_pols, self.nn_func.n_local_pts, self.nn_func.n_funcs_per_pol = g.shape

        phi = tf.tile(phi, [1, 1, self.nn_func.n_funcs_per_pol])
        phi = tf.transpose(phi, [0, 2, 1])
        phi = tf.reshape(phi, [-1, 1])
        g = tf.transpose(g, [0, 2, 1])
        g = tf.reshape(g, [-1, 1])

        self.nn_func.var_phi.assign(phi)
        self.nn_func.var_g.assign(g)

        if setup_n_derivatives == 1:
            phi_grad = tf.transpose(phi_grad, [0, 2, 1, 3])
            phi_grad = tf.reshape(phi_grad, [-1, 2])
            g_grad = tf.transpose(g_grad, [0, 2, 1, 3])
            g_grad = tf.reshape(g_grad, [-1, 2])

            # self.nn_func.var_phi_x.assign(phi_grad[:, 0:1])
            # self.nn_func.var_phi_y.assign(phi_grad[:, 1:2])
            # self.nn_func.var_g_x.assign(g_grad[:, 0:1])
            # self.nn_func.var_g_y.assign(g_grad[:, 1:2])
            self.nn_func.var_phi_grad.assign(phi_grad)
            self.nn_func.var_g_grad.assign(g_grad)

            if self.split_deriv:
                self.nn_grad.n_pols, self.nn_grad.n_local_pts, self.nn_grad.n_funcs_per_pol = [self.nn_func.n_pols,
                                                                                               self.nn_func.n_local_pts,
                                                                                               self.nn_func.n_funcs_per_pol]

    def save_model(self):
        self.flags = None
            
        zero_1d = tf.convert_to_tensor([0.], dtype=tf.float64)
        zero_2d = tf.convert_to_tensor([[0.]], dtype=tf.float64)

        self.nn_func.best_w.assign(zero_1d)
        self.nn_func.bfgs_regularization_grads.assign(zero_1d)
        self.nn_func.training_quad_w.assign(zero_2d)
        self.nn_func.var_g.assign(zero_2d)
        # self.nn_func.var_g_y.assign(zero_2d)
        # self.nn_func.var_g_xx.assign(zero_2d)
        # self.nn_func.var_g_yy.assign(zero_2d)
        self.nn_func.var_phi.assign(zero_2d)
        # self.nn_func.var_phi_y.assign(zero_2d)
        # self.nn_func.var_phi_xx.assign(zero_2d)
        # self.nn_func.var_phi_xy.assign(zero_2d)
        # self.nn_func.var_phi_yy.assign(zero_2d)
        # self.nn_func.x_vals = None
        # self.nn_func.y_vals = None
        self.nn_func.x_verts = None
        self.nn_func.y_verts = None
        self.nn_func.jac_invs = None

        if self.split_deriv:
            self.nn_grad.best_w.assign(zero_1d)
            self.nn_grad.bfgs_regularization_grads.assign(zero_1d)
            self.nn_grad.x_verts = None
            self.nn_grad.y_verts = None
            self.nn_grad.jac_invs = None

            # self.nn_grad_dx.best_w.assign(zero_1d)
            # self.nn_grad_dx.bfgs_regularization_grads.assign(zero_1d)
            # self.nn_grad_dx.training_quad_w.assign(zero_2d)
            # self.nn_grad_dx.var_g.assign(zero_2d)
            # self.nn_grad_dx.var_g_x.assign(zero_2d)
            # self.nn_grad_dx.var_g_y.assign(zero_2d)
            # self.nn_grad_dx.var_g_xx.assign(zero_2d)
            # self.nn_grad_dx.var_g_yy.assign(zero_2d)
            # self.nn_grad_dx.var_phi.assign(zero_2d)
            # self.nn_grad_dx.var_phi_x.assign(zero_2d)
            # self.nn_grad_dx.var_phi_y.assign(zero_2d)
            # self.nn_grad_dx.var_phi_xx.assign(zero_2d)
            # self.nn_grad_dx.var_phi_xy.assign(zero_2d)
            # self.nn_grad_dx.var_phi_yy.assign(zero_2d)
            # self.nn_grad_dx.x_vals = None
            # self.nn_grad_dx.y_vals = None
            # self.nn_grad_dx.x_verts = None
            # self.nn_grad_dx.y_verts = None
            # self.nn_grad_dx.jac_invs = None
            #
            # self.nn_grad_dy.best_w.assign(zero_1d)
            # self.nn_grad_dy.bfgs_regularization_grads.assign(zero_1d)
            # self.nn_grad_dy.training_quad_w.assign(zero_2d)
            # self.nn_grad_dy.var_g.assign(zero_2d)
            # self.nn_grad_dy.var_g_x.assign(zero_2d)
            # self.nn_grad_dy.var_g_y.assign(zero_2d)
            # self.nn_grad_dy.var_g_xx.assign(zero_2d)
            # self.nn_grad_dy.var_g_yy.assign(zero_2d)
            # self.nn_grad_dy.var_phi.assign(zero_2d)
            # self.nn_grad_dy.var_phi_x.assign(zero_2d)
            # self.nn_grad_dy.var_phi_y.assign(zero_2d)
            # self.nn_grad_dy.var_phi_xx.assign(zero_2d)
            # self.nn_grad_dy.var_phi_xy.assign(zero_2d)
            # self.nn_grad_dy.var_phi_yy.assign(zero_2d)
            # self.nn_grad_dy.x_vals = None
            # self.nn_grad_dy.y_vals = None
            # self.nn_grad_dy.x_verts = None
            # self.nn_grad_dy.y_verts = None
            # self.nn_grad_dy.jac_invs = None

        tf.print("saving weights on file ", self.name_storage)
        self.save_weights(self.name_storage + ".ckpt")
