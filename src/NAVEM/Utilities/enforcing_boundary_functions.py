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
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.path as plt_path
from pypolydim import gedim
from enum import Enum

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf


class BoundaryMethodType(Enum):
    line = 1
    segment = 2


class BubbleType(Enum):
    approximate_distance_function = 1
    product = 2


class EnforcingBoundary:
    def __init__(self, geometry_utilities: gedim.GeometryUtilities,
                 method_order: int,
                 vertices: NDArray[np.float64],
                 jac_per_pol: NDArray[np.float64],
                 method_type: BoundaryMethodType = BoundaryMethodType.segment,
                 bubble_type: BubbleType = BubbleType.approximate_distance_function):

        assert method_order == 1

        self.geometry_utilities = geometry_utilities
        self.method_order = method_order
        self.method_type = method_type
        self.bubble_type = bubble_type

        self.vertices = np.transpose(vertices, axes=[0, 2, 1])
        next_vertices = np.roll(self.vertices, shift=-1, axis=1)
        self.segments = np.concatenate([self.vertices, next_vertices], axis=2)
        self.num_vertices = vertices.shape[2]
        self.num_functions = self.num_vertices * self.method_order

        # compute tangent and normal for each edge
        self.lengths = np.linalg.norm(next_vertices - self.vertices, axis=2)
        self.tangents = next_vertices - self.vertices
        self.normals = self.tangents[:, :, [1, 0]] * np.array([-1, 1])
        self.normals /= np.linalg.norm(self.tangents, axis=2, keepdims=True)

        # projecting plane directions
        self.mid_points = 0.5 * (self.vertices + next_vertices)
        mid_p_normals = self.mid_points + self.normals
        z_a, z_b, z_c = [1, 0, 0.5]
        ux, uy, uz = [next_vertices[:, :, 0] - self.vertices[:, :, 0], next_vertices[:, :, 1] - self.vertices[:, :, 1],
                      z_b - z_a]
        vx, vy, vz = [mid_p_normals[:, :, 0] - self.vertices[:, :, 0], mid_p_normals[:, :, 1] - self.vertices[:, :, 1],
                      z_c - z_a]
        u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

        # projecting plane characterization
        point_3d = np.concatenate([self.vertices, np.ones_like(self.vertices[:, :, 0:1])], axis=2)
        self.proj_normal = np.transpose(np.array(u_cross_v), axes=[1, 2, 0])
        self.proj_d = -np.sum(point_3d * self.proj_normal, axis=2)

        # conversion to tensors
        self.jac_per_pol = jac_per_pol
        self.vertices = tf.convert_to_tensor(self.vertices, dtype=tf.float64)
        self.next_vertices = tf.convert_to_tensor(next_vertices, dtype=tf.float64)
        self.lengths = tf.convert_to_tensor(self.lengths, dtype=tf.float64)
        self.mid_points = tf.convert_to_tensor(self.mid_points, dtype=tf.float64)
        self.normals = tf.convert_to_tensor(self.normals, dtype=tf.float64)
        self.tangents = tf.convert_to_tensor(self.tangents, dtype=tf.float64)
        self.proj_normal = tf.convert_to_tensor(self.proj_normal, dtype=tf.float64)
        self.proj_d = tf.expand_dims(tf.convert_to_tensor(self.proj_d, dtype=tf.float64), 1)

    def compute_phi_k_line(self, xy: tf.Tensor) -> tf.Tensor:
        # Compute differences (x - x_v) and (y - y_v)
        diffs = xy[:, :, None, :] - self.vertices[:, None, :, :]  # Shape (num_points, num_vertices, 2)

        # Compute the final quantity: n_x * (x - x_v) + n_y * (y - y_v) with shape (num_points, num_vertices)
        # -eps is used to evaluate phi_k slightly inside the polygon in case (x,y) is on the boundary
        return tf.reduce_sum(self.normals[:, None, :, :] * diffs, axis=-1) - self.geometry_utilities.tolerance2_d()

    def compute_phi_k_segment(self, xy: tf.Tensor,
                              d_k: tf.Tensor,
                              return_t_k: bool = False) -> tf.Tensor | list[tf.Tensor]:

        exp_lengths = self.lengths[:, None, :]
        diff_wrt_mid_pts = xy[:, :, None, :] - self.mid_points[:, None, :, :]
        dist_from_mid_pts = tf.reduce_sum(tf.square(diff_wrt_mid_pts), axis=-1)

        t_k = 0.25 * exp_lengths - (1.0 / exp_lengths) * dist_from_mid_pts

        d_k_power2 = tf.square(d_k)
        z_k = tf.sqrt(t_k ** 2 + d_k_power2 ** 2)
        phi_k = tf.sqrt(d_k_power2 + 0.25 * (z_k - t_k) ** 2)

        if return_t_k:
            return [phi_k, t_k]
        else:
            return phi_k

    def projection(self, xy: tf.Tensor) -> tf.Tensor:
        # vectorized computation of: (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]
        return ((-tf.matmul(xy, tf.transpose(self.proj_normal[:, :, :2], perm=[0, 2, 1])) - self.proj_d) /
                self.proj_normal[:, tf.newaxis, :, 2])

    def transfinite_w(self, dist: tf.Tensor) -> tf.Tensor:
        numerator = tf.zeros(shape=(dist.shape[0], dist.shape[1], 0), dtype=tf.float64)
        # Compute all products of all terms except the i-th one
        for i in range(self.num_vertices):
            prod_all = tf.reduce_prod(dist[:, :, 0:i], axis=2, keepdims=True) * tf.reduce_prod(dist[:, :, i + 1:],
                                                                                               axis=2, keepdims=True)
            numerator = tf.concat([numerator, prod_all], axis=2)

        # Compute the denominator: sum of all terms where one dist[k] is removed
        # The numerator already excludes each element once
        denominator = tf.reduce_sum(numerator, axis=2, keepdims=True)  # Shape: (num_points, 1)

        # Compute the final ratio
        return numerator / denominator  # Shape: (num_points, num_vertices)

    def phi_and_g(self, xy: tf.Tensor) -> list[tf.Tensor]:
        proj = self.projection(xy)

        phi_k_line = self.compute_phi_k_line(xy)
        match self.method_type:
            case BoundaryMethodType.line:
                dist = phi_k_line
            case BoundaryMethodType.segment:
                dist = self.compute_phi_k_segment(xy, phi_k_line)
            case _:
                raise ValueError("Unknown boundary method type")

        transfinite_ws = self.transfinite_w(dist)

        prev_transfinite_w = tf.roll(transfinite_ws, shift=1, axis=2)
        prev_proj = tf.roll(proj, shift=1, axis=2)
        g = prev_transfinite_w * (1.0 - prev_proj) + transfinite_ws * proj

        match self.bubble_type:
            case BubbleType.approximate_distance_function:
                m = 2
                sum_inverse = tf.reduce_sum(tf.pow(dist, -tf.constant(m, dtype=dist.dtype)), axis=2, keepdims=True)
                phi = 1.0 / sum_inverse ** (1 / m)
            case BubbleType.product:
                phi = tf.reduce_prod(dist, axis=2, keepdims=True)
            case _:
                raise ValueError("Unknown bubble type")

        return [phi, g]

    def phi_and_g_and_grads(self, inputs: tf.Tensor) -> list[tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            phi, g = self.phi_and_g(inputs)

            g_slices = [g[:, :, v] for v in range(self.num_functions)]

        phi_grad = tape.gradient(phi, inputs)

        g_grad = [tape.gradient(g_slices[v], inputs) for v in range(self.num_functions)]
        g_grad = tf.stack(g_grad, 2)
        del tape

        return [phi, g, phi_grad, g_grad]

    def map_phi_and_g_grads(self, phi_grad: tf.Tensor, g_grad: tf.Tensor) -> list[tf.Tensor]:

        if self.jac_per_pol is None:
            phi_grad = tf.repeat(tf.expand_dims(phi_grad, axis=2), self.num_functions, axis=2)
            return [phi_grad, g_grad]
        else:
            g_grad_mapped = tf.einsum('pnvi,piov->pnvo', g_grad, self.jac_per_pol)
            phi_grad_mapped = tf.einsum('pni,piov->pnvo', phi_grad, self.jac_per_pol)
            return [phi_grad_mapped, g_grad_mapped]

    def map_phi_and_g_second_derivatives(self, phi_sec_ders: tf.Tensor, g_sec_ders: tf.Tensor) -> list[tf.Tensor]:

        if self.jac_per_pol is None:
            phi_sec_ders = tf.repeat(tf.expand_dims(phi_sec_ders, axis=2), self.num_functions, axis=2)
            return [phi_sec_ders, g_sec_ders]
        else:
            g_sec_ders_mapped = np.zeros(shape=g_sec_ders.numpy().shape)
            phi_sec_ders_mapped = np.zeros(shape=g_sec_ders.numpy().shape)

            for d1 in range(2):
                for d2 in range(2):
                    g_sec_ders_mapped[..., 0] += tf.einsum("pv,pv,psv->psv",
                                                           self.jac_per_pol[:, d1, 0, :],
                                                           self.jac_per_pol[:, d2, 0, :],
                                                           g_sec_ders[..., d1 + d2]
                                                           ).numpy()

                    g_sec_ders_mapped[..., 1] += tf.einsum("pv,pv,psv->psv",
                                                           self.jac_per_pol[:, d1, 0, :],
                                                           self.jac_per_pol[:, d2, 1, :],
                                                           g_sec_ders[..., d1 + d2]
                                                           ).numpy()

                    g_sec_ders_mapped[..., 2] += tf.einsum("pv,pv,psv->psv",
                                                           self.jac_per_pol[:, d1, 1, :],
                                                           self.jac_per_pol[:, d2, 1, :],
                                                           g_sec_ders[..., d1 + d2]
                                                           ).numpy()

                    phi_sec_ders_mapped[..., 0] += tf.einsum("pv,pv,ps->psv",
                                                             self.jac_per_pol[:, d1, 0, :],
                                                             self.jac_per_pol[:, d2, 0, :],
                                                             phi_sec_ders[..., d1 + d2]
                                                             ).numpy()

                    phi_sec_ders_mapped[..., 1] += tf.einsum("pv,pv,ps->psv",
                                                             self.jac_per_pol[:, d1, 0, :],
                                                             self.jac_per_pol[:, d2, 1, :],
                                                             phi_sec_ders[..., d1 + d2]
                                                             ).numpy()

                    phi_sec_ders_mapped[..., 2] += tf.einsum("pv,pv,ps->psv",
                                                             self.jac_per_pol[:, d1, 1, :],
                                                             self.jac_per_pol[:, d2, 1, :],
                                                             phi_sec_ders[..., d1 + d2]
                                                             ).numpy()
            return [tf.convert_to_tensor(phi_sec_ders_mapped, dtype=tf.float64),
                    tf.convert_to_tensor(g_sec_ders_mapped, dtype=tf.float64)]

    def phi_and_g_and_grads_and_second_derivatives(self, inputs: tf.Tensor) -> list[tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)

            phi, g, phi_grad, g_grad = self.phi_and_g_and_grads(inputs)

            phi_x, phi_y = [phi_grad[:, :, :1], phi_grad[:, :, 1:]]
            g_x, g_y = [g_grad[:, :, :, 0], g_grad[:, :, :, 1]]

            g_x_slices = [g_x[:, :, v] for v in range(self.num_functions)]
            g_y_slices = [g_y[:, :, v] for v in range(self.num_functions)]

        phi_x_ders = tape.gradient(phi_x, inputs)
        phi_y_ders = tape.gradient(phi_y, inputs)
        phi_sec_ders = tf.concat([phi_x_ders, phi_y_ders[:, :, 1:]], -1)

        g_x_ders = [tape.gradient(g_x_slices[v], inputs) for v in range(self.num_functions)]
        g_x_ders = tf.stack(g_x_ders, 2)
        g_y_ders = [tape.gradient(g_y_slices[v], inputs) for v in range(self.num_functions)]
        g_y_ders = tf.stack(g_y_ders, 2)
        g_sec_ders = tf.concat([g_x_ders, g_y_ders[:, :, :, 1:]], 3)

        del tape

        phi_grad_mapped, g_grad_mapped = self.map_phi_and_g_grads(phi_grad, g_grad)
        phi_sec_ders_mapped, g_sec_ders_mapped = self.map_phi_and_g_second_derivatives(phi_sec_ders, g_sec_ders)

        return [phi, g, phi_grad_mapped, g_grad_mapped, phi_sec_ders_mapped, g_sec_ders_mapped]

    def inside_pol(self, inputs: NDArray[np.float64] | tf.Tensor) -> tf.Tensor:
        inside = 0
        for p in range(self.vertices.shape[0]):
            polygon = self.vertices[p, :, :]
            path = plt_path.Path(polygon)
            curr_inside = path.contains_points(inputs[p, :, :])
            if p == 0:
                inside = np.expand_dims(curr_inside, 0)
            else:
                curr_inside = np.expand_dims(curr_inside, 0)
                inside = np.concatenate([inside, curr_inside], 0)
        return tf.convert_to_tensor(inside, dtype=tf.float64)

    def draw_function_one_edge(self, n: int, x: NDArray[np.float64], y: NDArray[np.float64],
                               xy: tf.Tensor, pol_id: int, edge_id: int):

        phi_k_line = self.compute_phi_k_line(xy)
        match self.method_type:
            case BoundaryMethodType.line:
                phi = phi_k_line
            case BoundaryMethodType.segment:
                phi = self.compute_phi_k_segment(xy, phi_k_line)
            case _:
                raise ValueError("Unknown boundary method type.")

        phi = phi[pol_id, :, edge_id]
        phi = phi.numpy().reshape((n, n))

        z = phi

        plt.contour(x, y, z, levels=20, linewidths=1, linestyles="solid", colors=["black"])
        plt.contourf(x, y, z, levels=200)
        cbar = plt.colorbar()
        for e in self.segments[pol_id, :, :]:
            plt.plot([e[0], e[2]], [e[1], e[3]], 'blue', linewidth=1)
        plt.plot([self.segments[pol_id, edge_id, 0], self.segments[pol_id, edge_id, 2]], [
            self.segments[pol_id, edge_id, 1], self.segments[pol_id, edge_id, 3]], 'red', linewidth=3)
        if edge_id < self.num_vertices - 1:
            plt.scatter(self.vertices[pol_id, edge_id:edge_id + 2, 0], self.vertices[pol_id, edge_id:edge_id + 2, 1],
                        c='red', marker='o', s=60)
        else:
            plt.scatter(self.vertices[pol_id, 0, 0], self.vertices[pol_id, 0, 1],
                        c='red', marker='o', s=60)
            plt.scatter(self.vertices[pol_id, self.num_vertices - 1, 0],
                        self.vertices[pol_id, self.num_vertices - 1, 1],
                        c='red', marker='o', s=60)
        cbar.ax.tick_params(labelsize=17)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def draw_function(self, n: int, x: NDArray[np.float64], y: NDArray[np.float64],
                      xy: tf.Tensor, pol_id: int, dof_id: int, draw_lifting: bool):

        phi, g = self.phi_and_g(xy)

        phi = phi[pol_id, :, :]
        g = g[pol_id, :, dof_id:dof_id + 1]
        inside = self.inside_pol(xy)[pol_id, :]

        lifting = g.numpy().reshape((n, n))
        phi = phi.numpy().reshape((n, n))

        lifting *= inside.numpy().reshape((n, n))
        phi *= inside.numpy().reshape((n, n))

        if draw_lifting:
            z = lifting
        else:
            z = phi

        z = np.ma.masked_where(inside.numpy().reshape((n, n)) == 0, z)

        plt.contour(x, y, z, levels=20, linewidths=1, linestyles="solid", colors=["black"])
        plt.contourf(x, y, z, levels=200)
        cbar = plt.colorbar()
        for e in self.segments[pol_id, :, :]:
            plt.plot([e[0], e[2]], [e[1], e[3]], 'red', linewidth=2)
        plt.scatter(self.vertices[pol_id, :, 0], self.vertices[pol_id, :, 1], c='red', marker='o', s=40)
        cbar.ax.tick_params(labelsize=17)
        plt.axis("equal")
        plt.show()

    def scatter_function(self, pol_id: int, dof_id: int, plot_all_edges: bool = True,
                         plot_inner_pts: bool = True, draw_lifting: bool = True):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')

        n = 51
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        x_all = np.expand_dims(np.tile(x, n), 1)
        y_all = np.expand_dims(np.repeat(y, n), 1)
        xy = np.concatenate([x_all, y_all], axis=1)
        xy_expanded = np.expand_dims(xy, 0)
        xy_expanded = np.tile(xy_expanded, [self.vertices.shape[0], 1, 1])
        inside_mask = (self.inside_pol(xy_expanded)[pol_id, :] == 1)
        x = x_all[inside_mask, 0]
        y = y_all[inside_mask, 0]

        new_x = np.zeros(shape=(0,))
        new_y = np.zeros(shape=(0,))
        linespace_01 = np.linspace(0, 1, n)
        pol_segments = self.segments[pol_id, :, :]
        for e in range(self.num_vertices):
            if (e != dof_id and e != (dof_id - 1) % self.num_vertices) or plot_all_edges:
                curr_new_x = pol_segments[e, 0] + linespace_01 * [pol_segments[e, 2] - pol_segments[e, 0]]
                curr_new_y = pol_segments[e, 1] + linespace_01 * [pol_segments[e, 3] - pol_segments[e, 1]]
                new_x = np.concatenate([new_x, curr_new_x])
                new_y = np.concatenate([new_y, curr_new_y])

        new_xy = np.concatenate([np.expand_dims(new_x, 1), np.expand_dims(new_y, 1)], axis=1)
        if plot_inner_pts:
            inner_xy = np.concatenate([np.expand_dims(x, 1), np.expand_dims(y, 1)], axis=1)
            new_xy = np.concatenate([new_xy, inner_xy], axis=0)

        new_xy = tf.expand_dims(new_xy, 0)
        new_xy = tf.concat([new_xy] * self.vertices.shape[0], axis=0)

        phi, g = self.phi_and_g(new_xy)
        phi = phi[pol_id, :]
        g = g[pol_id, :, dof_id:dof_id + 1]

        if draw_lifting:
            z = g
        else:
            z = phi

        ax.scatter(new_xy[pol_id, :, 0], new_xy[pol_id, :, 1], z[:, 0])
        plt.show()
