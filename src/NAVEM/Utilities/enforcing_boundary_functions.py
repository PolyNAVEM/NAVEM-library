import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pypolydim import gedim
import matplotlib.path as mpltPath
import os


class EnforcingBoundary:
    def __init__(self, geometry_utilities: gedim.GeometryUtilities):
        self.geometry_utilities = geometry_utilities

    def prepare_using_vertices(self, vertices, jac_per_pol = None, function_type = "line"):

        self.vertices = np.transpose(vertices, axes=[0, 2, 1])
        next_verts = np.roll(self.vertices, shift=-1, axis=1)
        self.function_type = function_type
        self.segments = np.concatenate([self.vertices, next_verts], axis=2)
        self.n_verts = vertices.shape[2]

        # compute tangent and normal for each edge
        self.lengths = np.linalg.norm(next_verts - self.vertices, axis=2)
        self.tangents = next_verts - self.vertices
        self.normals = self.tangents[:, :, [1, 0]] * np.array([-1, 1])
        self.normals /= np.linalg.norm(self.tangents, axis=2, keepdims=True)

        # projecting plane directions
        self.mid_points = 0.5 * (self.vertices + next_verts)
        mid_p_normals = self.mid_points + self.normals
        zA, zB, zC = [1, 0, 0.5]
        ux, uy, uz = [next_verts[:, :, 0] - self.vertices[:, :, 0], next_verts[:, :, 1] - self.vertices[:, :, 1], zB - zA]
        vx, vy, vz = [mid_p_normals[:, :, 0] - self.vertices[:, :, 0], mid_p_normals[:, :, 1] - self.vertices[:, :, 1], zC - zA]
        u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

        # projecting plane characterization
        point_3d = np.concatenate([self.vertices, np.ones_like(self.vertices[:, :, 0:1])], axis=2)
        self.proj_normal = np.transpose(np.array(u_cross_v), axes=[1, 2, 0])
        self.proj_d = -np.sum(point_3d * self.proj_normal, axis=2)

        # conversion to tensors
        self.jac_per_pol = jac_per_pol
        self.vertices = tf.convert_to_tensor(self.vertices, dtype=tf.float64)
        self.next_verts = tf.convert_to_tensor(next_verts, dtype=tf.float64)
        self.lengths = tf.convert_to_tensor(self.lengths, dtype=tf.float64)
        self.mid_points = tf.convert_to_tensor(self.mid_points, dtype=tf.float64)
        self.normals = tf.convert_to_tensor(self.normals, dtype=tf.float64)
        self.tangents = tf.convert_to_tensor(self.tangents, dtype=tf.float64)
        self.proj_normal = tf.convert_to_tensor(self.proj_normal, dtype=tf.float64)
        self.proj_d = tf.expand_dims(tf.convert_to_tensor(self.proj_d, dtype=tf.float64), 1)

    def compute_phi_k_line(self, xy):
        # Compute differences (xP - xV) and (yP - yV)
        diffs = xy[:, :, None, :] - self.vertices[:, None, :, :]  # Shape (Npts, Nvrts, 2)

        # Compute the final quantity: nxV * (xP - xV) + nyV * (yP - yV) with shape (Npts, Nvrts)
        # -eps serve a valutare sempre leggermente dentro al poligono anche quando si valuta sul bordo
        # return tf.reduce_sum(abs(self.normals[:, None, :, :] * diffs), axis=-1) - self.geometry_utilities.geometry_tol_2d
        return tf.reduce_sum(self.normals[:, None, :, :] * diffs, axis=-1) - self.geometry_utilities.tolerance2_d()

    def compute_phi_k_segment(self, xy, d_k, return_t_k = False):
        exp_lengths = self.lengths[:, None, :]
        diff_wrt_mid_pts = xy[:, :, None, :] - self.mid_points[:, None, :, :]
        dist_from_mid_pts = tf.reduce_sum(tf.square(diff_wrt_mid_pts), axis=-1)


        # f = (1 / exp_lengths) * (xy_vA[:, :, :, 0] * vA_vB[:, :, :, 1] - xy_vA[:, :, :, 1] * vA_vB[:, :, :, 0])
        t_k = 0.25 * exp_lengths - (1.0 / exp_lengths) * dist_from_mid_pts

        # exp_xy = xy[:, :, None, :] # half-line
        # exp_verts = self.vertices[:, None, :, :]
        # xy_vA = exp_xy - exp_verts
        # vA_vB = self.next_verts[:, None, :, :] - exp_verts
        # t = (1 / exp_lengths) * tf.reduce_sum(xy_vA*vA_vB, axis=-1)

        d_k_power2 = d_k ** 2
        z_k = tf.sqrt(t_k ** 2 + d_k_power2 ** 2)
        phi_k = tf.sqrt(d_k_power2 + 0.25 * (z_k - t_k) ** 2)

        if return_t_k:
            return phi_k, t_k
        else:
            return phi_k

    def compute_phi_k_half_line_left(self, xy, d_k):
        exp_lengths = self.lengths[:, None, :]
        inv_exp_lengths = 1.0 / exp_lengths

        t_k = (xy[:, :, None, :] - self.next_verts[:, None, :, :])
        t_k = - inv_exp_lengths * tf.reduce_sum(self.tangents[:, None, :, :] * t_k, axis=-1)

        z_k = tf.sqrt(t_k ** 2 + d_k ** 4)
        phi_k = tf.sqrt(d_k ** 2 + 0.25 * (z_k - t_k) ** 2)

        return phi_k

    def compute_phi_k_half_line_right(self, xy, d_k):
        exp_lengths = self.lengths[:, None, :]
        inv_exp_lengths = 1.0 / exp_lengths

        t_k = (xy[:, :, None, :] - self.vertices[:, None, :, :])
        t_k = inv_exp_lengths * tf.reduce_sum(self.tangents[:, None, :, :] * t_k, axis=-1)

        z_k = tf.sqrt(t_k ** 2 + d_k ** 4)
        phi_k = tf.sqrt(d_k ** 2 + 0.25 * (z_k - t_k) ** 2)

        return phi_k

    def projection(self, xy):
        # Voglio vettorizzare questo: (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]
        return (-tf.matmul(xy, tf.transpose(self.proj_normal[:, :, :2], perm=[0, 2, 1])) - self.proj_d) / self.proj_normal[:, tf.newaxis, :, 2]

    def transfinite_w(self, dist):
        # # Compute the product of all distances along axis 1 (for each row)
        # prod_all = tf.reduce_prod(dist, axis=2, keepdims=True)  # Shape: (Npts, 1)
        #
        # # Compute the numerator: Exclude dist[i] by dividing by it
        # numerator = prod_all / dist  # Shape: (Npts, Nvrts)
        numerator = tf.zeros(shape=(dist.shape[0], dist.shape[1], 0), dtype=tf.float64)
        for i in range(self.n_verts):
            prod_all = tf.reduce_prod(dist[:, :, 0:i], axis=2, keepdims=True) * tf.reduce_prod(dist[:, :, i+1:], axis=2, keepdims=True)
            numerator = tf.concat([numerator, prod_all], axis=2)

        # Compute the denominator: Sum of all terms where one dist[k] is removed
        # The numerator already excludes each element once
        denominator = tf.reduce_sum(numerator, axis=2, keepdims=True)  # Shape: (Npts, 1)

        # Compute final ratio
        return numerator / denominator  # Shape: (Npts, Nvrts)

    def phi_and_g(self, xy, reshape_for_net=False, method_type=1, bubble_type=1):
        proj = self.projection(xy)

        phi_k_line = self.compute_phi_k_line(xy)
        if method_type == 0:
            dist = phi_k_line
        elif method_type == 1:
            dist = self.compute_phi_k_segment(xy, phi_k_line)
        elif method_type == 2:
            dist = self.compute_phi_k_half_line_right(xy, phi_k_line)
        elif method_type == 3:
            dist = self.compute_phi_k_half_line_left(xy, phi_k_line)
        elif method_type == 4:
            phi_k_half_line_left = self.compute_phi_k_half_line_left(xy, phi_k_line)
            phi_k_half_line_right = self.compute_phi_k_half_line_right(xy, phi_k_line)
            phi_k_segment = self.compute_phi_k_segment(xy, phi_k_line)

            dist = (phi_k_line * (self.select_b_type[:, :, :] == 0) + phi_k_segment * (self.select_b_type[:, :, :] == 1)
                    + phi_k_half_line_left * (self.select_b_type[:, :, :] == 2)
                    + phi_k_half_line_right * (self.select_b_type[:, :, :] == 3))

        transf_ws = self.transfinite_w(dist)

        prev_transf_w = tf.roll(transf_ws, shift=1, axis=2)
        prev_proj = tf.roll(proj, shift=1, axis=2)
        g = prev_transf_w * (1.0-prev_proj) + transf_ws * proj

        if bubble_type == 1:
            # ADF (derivata normale unitaria)
            m = 2
            sum_invers = tf.reduce_sum(1.0 / (dist**m), axis=2, keepdims=True)
            phi = 1.0 / sum_invers**(1/m)
        elif bubble_type == 2:
            # CLASSICA BOLLA (rischia di avere derivate normali molto piccole vicino ad alcuni bordi/vertici)
            phi = tf.reduce_prod(dist, axis=2, keepdims=True)

        return phi, g

    def phi_and_g_and_grads(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            phi, g = self.phi_and_g(inputs)

            g_slices = [g[:, :, v] for v in range(self.n_verts)]

        phi_grad = tape.gradient(phi, inputs)

        g_grad = [tape.gradient(g_slices[v], inputs) for v in range(self.n_verts)]
        g_grad = tf.stack(g_grad, 2)
        del tape

        return phi, g, phi_grad, g_grad

    def map_phi_and_g_grads(self, phi_grad, g_grad):

        if self.jac_per_pol is None:
            phi_grad = np.repeat(np.expand_dims(phi_grad, axis=2), self.n_verts, axis=2)
            return phi_grad, g_grad
        else:
            g_grad_mapped = np.einsum('pnvi,piov->pnvo', g_grad, self.jac_per_pol)
            phi_grad_mapped = np.einsum('pni,piov->pnvo', phi_grad, self.jac_per_pol)

            return phi_grad_mapped, g_grad_mapped

    def map_phi_and_g_second_derivatives(self, phi_sec_ders, g_sec_ders):
        tf.print("\n Stai usando EnforcingBoundary.map_phi_and_g_second_derivatives() per calcolare la loss, questa funzione è stralenta, se proprio ti "
                 "serve usarla riscrivila come EnforcingBoundary.map_phi_and_g_grads() con np.einsum()\n")


        if self.jac_per_pol is None:
            phi_sec_ders = np.repeat(np.expand_dims(phi_sec_ders, axis=2), self.n_verts, axis=2)
            return phi_sec_ders, g_sec_ders
        else:
            g_sec_ders_mapped = np.zeros(shape=g_sec_ders.shape)
            phi_sec_ders_mapped = np.zeros(shape=g_sec_ders.shape)

            for p in range(g_sec_ders.shape[0]):
                for v in range(g_sec_ders.shape[2]):
                    for d1 in range(2):
                        for d2 in range(2):
                            g_sec_ders_mapped[p, :, v, 0] += self.jac_per_pol[p, d1, 0, v] * self.jac_per_pol[p, d2, 0, v] * g_sec_ders[p, :, v, d1 + d2]
                            g_sec_ders_mapped[p, :, v, 1] += self.jac_per_pol[p, d1, 0, v] * self.jac_per_pol[p, d2, 1, v] * g_sec_ders[p, :, v, d1 + d2]
                            g_sec_ders_mapped[p, :, v, 2] += self.jac_per_pol[p, d1, 1, v] * self.jac_per_pol[p, d2, 1, v] * g_sec_ders[p, :, v, d1 + d2]

                            phi_sec_ders_mapped[p, :, v, 0] += self.jac_per_pol[p, d1, 0, v] * self.jac_per_pol[p, d2, 0, v] * phi_sec_ders[p, :, d1 + d2]
                            phi_sec_ders_mapped[p, :, v, 1] += self.jac_per_pol[p, d1, 0, v] * self.jac_per_pol[p, d2, 1, v] * phi_sec_ders[p, :, d1 + d2]
                            phi_sec_ders_mapped[p, :, v, 2] += self.jac_per_pol[p, d1, 1, v] * self.jac_per_pol[p, d2, 1, v] * phi_sec_ders[p, :, d1 + d2]

            return phi_sec_ders_mapped, g_sec_ders_mapped

    def phi_and_g_and_grads_and_second_derivatives(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)

            phi, g, phi_grad, g_grad = self.phi_and_g_and_grads(inputs)

            phi_x, phi_y = [phi_grad[:, :, :1], phi_grad[:, :, 1:]]
            g_x, g_y = [g_grad[:, :, :, 0], g_grad[:, :, :, 1]]

            g_x_slices = [g_x[:, :, v] for v in range(self.n_verts)]
            g_y_slices = [g_y[:, :, v] for v in range(self.n_verts)]

        phi_x_ders = tape.gradient(phi_x, inputs)
        phi_y_ders = tape.gradient(phi_y, inputs)
        phi_sec_ders = tf.concat([phi_x_ders, phi_y_ders[:, :, 1:]], -1)

        g_x_ders = [tape.gradient(g_x_slices[v], inputs) for v in range(self.n_verts)]
        g_x_ders = tf.stack(g_x_ders, 2)
        g_y_ders = [tape.gradient(g_y_slices[v], inputs) for v in range(self.n_verts)]
        g_y_ders = tf.stack(g_y_ders, 2)
        g_sec_ders = tf.concat([g_x_ders, g_y_ders[:, :, :, 1:]], 3)

        del tape

        phi_grad_mapped, g_grad_mapped = self.map_phi_and_g_grads(phi_grad, g_grad)
        phi_sec_ders_mapped, g_sec_ders_mapped = self.map_phi_and_g_second_derivatives(phi_sec_ders, g_sec_ders)

        return phi, g, phi_grad_mapped, g_grad_mapped, phi_sec_ders_mapped, g_sec_ders_mapped

    def inside_pol(self, inputs):
        for p in range(self.vertices.shape[0]):
            polygon = self.vertices[p, :, :]
            path = mpltPath.Path(polygon)
            curr_inside = path.contains_points(inputs[p, :, :])
            if p == 0:
                inside = np.expand_dims(curr_inside, 0)
            else:
                curr_inside = np.expand_dims(curr_inside, 0)
                inside = np.concatenate([inside, curr_inside], 0)
        return tf.convert_to_tensor(inside, dtype=tf.float64)

    def draw_function_one_edge(self, N, x, y, xy, pol_id, edge_id, method_type, file_path: str):

        phi_k_line = self.compute_phi_k_line(xy)
        if method_type == 0:
            phi = phi_k_line
        elif method_type == 1:
            phi = self.compute_phi_k_segment(xy, phi_k_line)
        elif method_type == 2:
            phi = self.compute_phi_k_half_line_right(xy, phi_k_line)
        elif method_type == 3:
            phi = self.compute_phi_k_half_line_left(xy, phi_k_line)

        phi = phi[pol_id, :, edge_id]
        phi = phi.numpy().reshape((N, N))

        z = phi

        plt.contour(x, y, z, levels=20, linewidths=1, linestyles="solid", colors=["black"])
        plt.contourf(x, y, z, levels=200)
        cbar = plt.colorbar()
        # for e in self.segments[pol_id, :, :]:
        #    plt.plot([e[0], e[2]], [e[1], e[3]], 'blue', linewidth=1)
        plt.plot([self.segments[pol_id, edge_id, 0], self.segments[pol_id, edge_id, 2]], [
                 self.segments[pol_id, edge_id, 1], self.segments[pol_id, edge_id, 3]], 'red', linewidth=3)
        # plt.scatter(self.vertices[pol_id, :, 0], self.vertices[pol_id, :, 1], c='blue', marker='o', s=50)
        if edge_id < self.n_verts - 1:
            plt.scatter(self.vertices[pol_id, edge_id:edge_id + 2, 0], self.vertices[pol_id, edge_id:edge_id + 2, 1],
                        c='red', marker='o', s=60)
        else:
            plt.scatter(self.vertices[pol_id, 0, 0], self.vertices[pol_id, 0, 1],
                        c='red', marker='o', s=60)
            plt.scatter(self.vertices[pol_id, self.n_verts-1, 0], self.vertices[pol_id, self.n_verts-1, 1],
                        c='red', marker='o', s=60)
        cbar.ax.tick_params(labelsize=17)
        plt.axis("equal")
        plt.axis("off")

        plt.savefig(file_path + "/phi0_edge_" + str(edge_id) + "_mt_" + str(method_type) +
                    "_pol_id_" + str(pol_id) + "_nv_" + str(self.n_verts) + ".png",
                    bbox_inches='tight')
        plt.show()

    def draw_trimming_function_one_edge(self, N, x, y, xy, pol_id, edge_id, file_path: str):

        phi_k_line = self.compute_phi_k_line(xy)
        _, t_k = self.compute_phi_k_segment(xy, phi_k_line, return_t_k=True)

        t_k = t_k[pol_id, :, edge_id]
        t = t_k.numpy().reshape((N, N))
        z = t

        plt.contour(x, y, z, levels=20, linewidths=1, linestyles="solid", colors=["black"])
        plt.contourf(x, y, z, levels=200)
        cbar = plt.colorbar()
        # for e in self.segments[pol_id, :, :]:
        #    plt.plot([e[0], e[2]], [e[1], e[3]], 'blue', linewidth=1)
        plt.plot([self.segments[pol_id, edge_id, 0], self.segments[pol_id, edge_id, 2]], [
            self.segments[pol_id, edge_id, 1], self.segments[pol_id, edge_id, 3]], 'red', linewidth=3)
        # plt.scatter(self.vertices[pol_id, :, 0], self.vertices[pol_id, :, 1], c='blue', marker='o', s=50)
        if edge_id < self.n_verts - 1:
            plt.scatter(self.vertices[pol_id, edge_id:edge_id + 2, 0], self.vertices[pol_id, edge_id:edge_id + 2, 1],
                        c='red', marker='o', s=60)
        else:
            plt.scatter(self.vertices[pol_id, 0, 0], self.vertices[pol_id, 0, 1],
                        c='red', marker='o', s=60)
            plt.scatter(self.vertices[pol_id, self.n_verts - 1, 0], self.vertices[pol_id, self.n_verts - 1, 1],
                        c='red', marker='o', s=60)
        cbar.ax.tick_params(labelsize=17)
        plt.axis("equal")
        plt.axis("off")

        plt.savefig(file_path + "/trim_edge_" + str(edge_id) +
                    "_pol_id_" + str(pol_id) + "_nv_" + str(self.n_verts) + ".png",
                    bbox_inches='tight')
        plt.show()

    def draw_function(self, N, x, y, xy, pol_id, dof_id, draw_lifting, file_path: str, method_type=1, bubble_type=1):


        phi, g = self.phi_and_g(xy, method_type=method_type, bubble_type=bubble_type)

        phi = phi[pol_id, :, :]
        g = g[pol_id, :, dof_id:dof_id+1]
        inside = self.inside_pol(xy)[pol_id, :]

        lifting = g.numpy().reshape((N, N))
        # print(phi.shape)
        phi = phi.numpy().reshape((N, N))
        lifting *= inside.numpy().reshape((N, N))
        phi *= inside.numpy().reshape((N, N))

        if draw_lifting:
            z = lifting
        else:
            z = phi

        z = np.ma.masked_where(inside.numpy().reshape((N, N)) == 0, z)

        plt.contour(x, y, z, levels=20, linewidths=1, linestyles="solid", colors=["black"])
        plt.contourf(x, y, z, levels=200)
        cbar = plt.colorbar()
        for e in self.segments[pol_id, :, :]:
            plt.plot([e[0], e[2]], [e[1], e[3]], 'red', linewidth=4)
        plt.scatter(self.vertices[pol_id, :, 0], self.vertices[pol_id, :, 1], c='red', marker='o', s=100)
        cbar.ax.tick_params(labelsize=17)
        plt.axis("equal")

        plt.savefig(file_path + "/phi0_mt_" + str(method_type) + "_bt_"
                    + str(bubble_type) + "_pol_id_" + str(pol_id) + "_nv_" + str(self.n_verts) + ".png", bbox_inches='tight')
        plt.show()

    def draw_sec_der_function(self, N, x, y, xy, pol_id, dof_id, draw_lifting):

        # phi, g = self.phi_and_g(xy)
        # phi, g, phi_grad, g_grad = self.phi_and_g_and_grads(xy)
        phi, g, phi_grad, g_grad, phi_sec_ders, g_sec_ders = self.phi_and_g_and_grads_and_second_derivatives(xy)

        # norma del gradiente
        # phi = np.sqrt(phi_grad[pol_id, :, dof_id:dof_id+1, 0]**2 + phi_grad[pol_id, :, dof_id:dof_id+1, 1]**2)
        # g = np.sqrt(g_grad[pol_id, :, dof_id:dof_id+1, 0]**2 + g_grad[pol_id, :, dof_id:dof_id+1, 1]**2)

        # laplaciano
        phi = phi_sec_ders[pol_id, :, dof_id:dof_id+1, 0] + phi_sec_ders[pol_id, :, dof_id:dof_id+1, 2]
        g = g_sec_ders[pol_id, :, dof_id:dof_id+1, 0] + g_sec_ders[pol_id, :, dof_id:dof_id+1, 2]
        inside = self.inside_pol(xy)[pol_id, :]

        lifting = g.reshape((N, N))
        phi = phi.reshape((N, N))
        lifting *= inside.numpy().reshape((N, N))
        phi *= inside.numpy().reshape((N, N))

        if draw_lifting:
            z = lifting
        else:
            z = phi

        plt.contour(x, y, z, levels=15, linewidths=1, linestyles="solid", colors=["black"])
        plt.contourf(x, y, z, levels=150)
        cbar = plt.colorbar()
        for e in self.segments[pol_id, :, :]:
            plt.plot([e[0], e[2]], [e[1], e[3]], 'red', linewidth=1.5)
        cbar.ax.tick_params(labelsize=17)
        plt.axis("equal")
        plt.show()

    def scatter_border(self, pol_id, dof_id, method_type, plot_all_edges=True, plot_inner_pts=True, draw_lifting=True):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')

        N = 51
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        xall = np.expand_dims(np.tile(x, N), 1)
        yall = np.expand_dims(np.repeat(y, N), 1)
        xy = np.concatenate([xall, yall], axis=1)
        xy_expanded = np.expand_dims(xy,0)
        xy_expanded = np.tile(xy_expanded, [self.vertices.shape[0], 1, 1])
        inside_mask = (self.inside_pol(xy_expanded)[pol_id, :] == 1)
        x = xall[inside_mask, 0]
        y = yall[inside_mask, 0]

        new_x = np.zeros(shape=(0,))
        new_y = np.zeros(shape=(0,))
        linsp01 = np.linspace(0, 1, N)
        pol_segments = self.segments[pol_id, :, :]
        for e in range(self.n_verts):
            if (e != dof_id and e != (dof_id-1) % self.n_verts) or plot_all_edges:
                curr_new_x = pol_segments[e, 0] + linsp01 * [pol_segments[e, 2] - pol_segments[e, 0]]
                curr_new_y = pol_segments[e, 1] + linsp01 * [pol_segments[e, 3] - pol_segments[e, 1]]
                new_x = np.concatenate([new_x, curr_new_x])
                new_y = np.concatenate([new_y, curr_new_y])

        new_xy = np.concatenate([np.expand_dims(new_x, 1), np.expand_dims(new_y, 1)], axis=1)
        if plot_inner_pts:
            inner_xy = np.concatenate([np.expand_dims(x, 1), np.expand_dims(y, 1)], axis=1)
            new_xy = np.concatenate([new_xy, inner_xy], axis=0)

        new_xy = tf.expand_dims(new_xy, 0)
        new_xy = tf.concat([new_xy] * self.vertices.shape[0], axis=0)

        n_points_per_pol = new_xy.shape[1]
        self.select_b_type = np.zeros(shape=(self.vertices.shape[0], n_points_per_pol, self.vertices.shape[2]))
        for c in range(self.vertices.shape[0]):
            self.select_b_type[c, :, :] = np.repeat(np.expand_dims(np.array([method_type]), axis=0),
                                                    n_points_per_pol, axis=0)

        phi, g = self.phi_and_g(new_xy)
        phi = phi[pol_id, :]
        g = g[pol_id, :, dof_id:dof_id+1]

        if draw_lifting:
            z = g
        else:
            z = phi

        ax.scatter(new_xy[pol_id, :, 0], new_xy[pol_id, :, 1], z[:, 0])
        plt.show()


