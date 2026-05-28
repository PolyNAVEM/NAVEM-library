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
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.path as plt_path
from pypolydim import gedim
from enum import Enum

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf


class BoundaryMethodType(Enum):
    segment = 1
    line = 2
    wachspress = 3


class BubbleType(Enum):
    approximate_distance_function = 1
    product = 2


class EnforcingBoundary:

    vertices: tf.Tensor
    num_vertices: int
    num_boundary_functions: int

    next_vertices: tf.Tensor
    edge_tangents: tf.Tensor
    edge_normals: tf.Tensor
    edge_lengths: tf.Tensor
    edge_mid_points: tf.Tensor

    def __init__(self,
                 geometry_utilities: gedim.GeometryUtilities,
                 method_order: int,
                 boundary_method_type_adfs: BoundaryMethodType = BoundaryMethodType.segment,
                 boundary_method_type_g: BoundaryMethodType = BoundaryMethodType.segment,
                 bubble_type: BubbleType = BubbleType.approximate_distance_function):

        self.geometry_utilities = geometry_utilities
        self.method_order = method_order
        self.boundary_method_type_adfs = boundary_method_type_adfs
        self.boundary_method_type_g = boundary_method_type_g
        self.bubble_type = bubble_type

    def initialize_boundary_properties(self, vertices: tf.Tensor):

        """
        Parameters:
            vertices: tf.Tensor num_polygons x num_vertices x 2
        """

        self.vertices = vertices
        self.num_vertices = self.vertices.shape[1]
        self.num_boundary_functions = self.num_vertices * self.method_order
        self.next_vertices = tf.roll(vertices, shift=-1, axis=1)
        self.edge_tangents = self.next_vertices - self.vertices

        self.edge_lengths = tf.linalg.norm(self.next_vertices - self.vertices, axis=2)
        self.edge_mid_points = 0.5 * (self.vertices + self.next_vertices)

        self.edge_normals = tf.concat([
            -self.edge_tangents[:, :, 1:2],
            self.edge_tangents[:, :, 0:1]], axis=2)
        self.edge_normals = tf.linalg.l2_normalize(self.edge_normals, axis=2)

    def compute_phi_k_line(self, xy: tf.Tensor) -> tf.Tensor:

        """
        Parameters:
            xy: NDArray[np.float64] num_polygons x num_points_per_polygon x 2
        """

        # Compute differences (x - x_v) and (y - y_v)
        diffs = xy[:, :, None, :] - self.vertices[:, None, :, :]  # Shape (num_points, num_vertices, 2)

        # Compute the final quantity: n_x * (x - x_v) + n_y * (y - y_v) with shape (num_points, num_vertices)
        # -eps is used to evaluate phi_k slightly inside the polygon in case (x,y) is on the boundary
        return tf.reduce_sum(self.edge_normals[:, None, :, :] * diffs, axis=-1) - self.geometry_utilities.tolerance2_d()

    def compute_phi_k_segment(self,
                              xy: tf.Tensor,
                              phi_k_line: tf.Tensor) -> tf.Tensor:

        """
        Parameters:
            xy: tf.Tensor num_polygons x num_points_per_polygon x 2
            phi_k_line: tf.Tensor output of self.compute_phi_k_line
        """

        exp_lengths = self.edge_lengths[:, None, :]
        diff_wrt_mid_pts = xy[:, :, None, :] - self.edge_mid_points[:, None, :, :]
        dist_from_mid_pts = tf.reduce_sum(tf.square(diff_wrt_mid_pts), axis=-1)

        t_k = 0.25 * exp_lengths - (1.0 / exp_lengths) * dist_from_mid_pts

        d_k_power2 = tf.square(phi_k_line)
        z_k = tf.sqrt(t_k ** 2 + d_k_power2 ** 2)
        phi_k = tf.sqrt(d_k_power2 + 0.25 * (z_k - t_k) ** 2)

        return phi_k

    def projection(self, xy: tf.Tensor) -> tf.Tensor:

        mid_p_normals = self.edge_mid_points + self.edge_normals

        z_a = tf.constant(1.0, dtype=tf.float64)
        z_b = tf.constant(0.0, dtype=tf.float64)
        z_c = tf.constant(0.5, dtype=tf.float64)

        # u = next_vertices - vertices
        ux = self.next_vertices[:, :, 0] - self.vertices[:, :, 0]
        uy = self.next_vertices[:, :, 1] - self.vertices[:, :, 1]
        uz = z_b - z_a

        # v = mid_p_normals - vertices
        vx = mid_p_normals[:, :, 0] - self.vertices[:, :, 0]
        vy = mid_p_normals[:, :, 1] - self.vertices[:, :, 1]
        vz = z_c - z_a

        # cross product u × v
        u_cross_v_x = uy * vz - uz * vy
        u_cross_v_y = uz * vx - ux * vz
        u_cross_v_z = ux * vy - uy * vx

        proj_normal = tf.stack([u_cross_v_x, u_cross_v_y, u_cross_v_z], axis=-1)

        point_3d = tf.concat([self.vertices, tf.ones_like(self.vertices[:, :, 0:1])], axis=2)
        proj_d = -tf.reduce_sum(point_3d * proj_normal, axis=-1)
        proj_d = tf.expand_dims(proj_d, axis=1)

        # vectorized computation of: (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]
        return ((-tf.matmul(xy, tf.transpose(proj_normal[:, :, :2], perm=[0, 2, 1])) - proj_d) /
                proj_normal[:, tf.newaxis, :, 2])

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

    def compute_g(self, xy: tf.Tensor) -> tf.Tensor:

        phi_k_line = self.compute_phi_k_line(xy)
        dist = tf.constant([], tf.float64)
        match self.boundary_method_type_g:
            case BoundaryMethodType.line:
                dist = phi_k_line
            case BoundaryMethodType.segment:
                dist = self.compute_phi_k_segment(xy, phi_k_line)
            case _:
                raise ValueError("Unknown boundary method type")

        proj = self.projection(xy)
        transfinite_ws = self.transfinite_w(dist)
        prev_transfinite_w = tf.roll(transfinite_ws, shift=1, axis=2)
        prev_proj = tf.roll(proj, shift=1, axis=2)

        g = prev_transfinite_w * (1.0 - prev_proj) + transfinite_ws * proj

        return g

    def compute_g_and_grads(self, xy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xy)
            g = self.compute_g(xy)

            g_slices = [g[:, :, v] for v in range(self.num_boundary_functions)]

        g_grad = [tape.gradient(g_slices[v], xy) for v in range(self.num_boundary_functions)]
        g_grad = tf.stack(g_grad, 2)

        del tape

        return g, g_grad

    @staticmethod
    def map_g_grads(g_grad: tf.Tensor, jac_per_pol: tf.Tensor) -> tf.Tensor:

        g_grad_mapped = tf.einsum('pnvi,piov->pnvo', g_grad, jac_per_pol)
        return g_grad_mapped

    @staticmethod
    def map_g_second_derivatives(g_sec_ders: tf.Tensor, jac_per_pol: tf.Tensor) -> tf.Tensor:

        g_sec_ders_mapped = np.zeros(shape=g_sec_ders.numpy().shape)

        for d1 in range(2):
            for d2 in range(2):
                g_sec_ders_mapped[..., 0] += tf.einsum("pv,pv,psv->psv",
                                                       jac_per_pol[:, d1, 0, :],
                                                       jac_per_pol[:, d2, 0, :],
                                                       g_sec_ders[..., d1 + d2]
                                                       ).numpy()

                g_sec_ders_mapped[..., 1] += tf.einsum("pv,pv,psv->psv",
                                                       jac_per_pol[:, d1, 0, :],
                                                       jac_per_pol[:, d2, 1, :],
                                                       g_sec_ders[..., d1 + d2]
                                                       ).numpy()

                g_sec_ders_mapped[..., 2] += tf.einsum("pv,pv,psv->psv",
                                                       jac_per_pol[:, d1, 1, :],
                                                       jac_per_pol[:, d2, 1, :],
                                                       g_sec_ders[..., d1 + d2]
                                                       ).numpy()

        return tf.convert_to_tensor(g_sec_ders_mapped, dtype=tf.float64)

    def compute_g_and_grads_and_second_derivatives(self, xy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xy)
            g, g_grad = self.compute_g_and_grads(xy)

            g_x, g_y = [g_grad[:, :, :, 0], g_grad[:, :, :, 1]]

            g_x_slices = [g_x[:, :, v] for v in range(self.num_boundary_functions)]
            g_y_slices = [g_y[:, :, v] for v in range(self.num_boundary_functions)]

        g_x_ders = [tape.gradient(g_x_slices[v], xy) for v in range(self.num_boundary_functions)]
        g_x_ders = tf.stack(g_x_ders, 2)
        g_y_ders = [tape.gradient(g_y_slices[v], xy) for v in range(self.num_boundary_functions)]
        g_y_ders = tf.stack(g_y_ders, 2)
        g_sec_ders = tf.concat([g_x_ders, g_y_ders[:, :, :, 1:]], 3)

        del tape

        return g, g_grad, g_sec_ders

    def compute_adfs(self, xy: tf.Tensor) -> tf.Tensor:

        phi_k_line = self.compute_phi_k_line(xy)
        dist = tf.constant([], tf.float64)
        match self.boundary_method_type_adfs:
            case BoundaryMethodType.line:
                dist = phi_k_line
            case BoundaryMethodType.segment:
                dist = self.compute_phi_k_segment(xy, phi_k_line)
            case _:
                raise ValueError("Unknown boundary method type")

        phi = None
        match self.bubble_type:
            case BubbleType.approximate_distance_function:
                m = 2
                sum_inverse = tf.reduce_sum(tf.pow(dist, -tf.constant(m, dtype=dist.dtype)), axis=2, keepdims=True)
                phi = 1.0 / sum_inverse ** (1 / m)
            case BubbleType.product:
                phi = tf.reduce_prod(dist, axis=2, keepdims=True)
            case _:
                raise ValueError("Unknown bubble type")

        return phi

    @staticmethod
    def map_adf_grads(phi_grad: tf.Tensor, jac_per_pol: tf.Tensor) -> tf.Tensor:
        phi_grad_mapped = tf.einsum('pni,piov->pnvo', phi_grad, jac_per_pol)
        return phi_grad_mapped

    @staticmethod
    def map_adf_second_derivatives(phi_sec_ders: tf.Tensor, jac_per_pol: tf.Tensor) -> tf.Tensor:

        phi_sec_ders_mapped = np.zeros([phi_sec_ders.shape[0], phi_sec_ders.shape[1], jac_per_pol.shape[-1], 3])

        for d1 in range(2):
            for d2 in range(2):
                phi_sec_ders_mapped[..., 0] += tf.einsum("pv,pv,ps->psv",
                                                         jac_per_pol[:, d1, 0, :],
                                                         jac_per_pol[:, d2, 0, :],
                                                         phi_sec_ders[..., d1 + d2]
                                                         ).numpy()

                phi_sec_ders_mapped[..., 1] += tf.einsum("pv,pv,ps->psv",
                                                         jac_per_pol[:, d1, 0, :],
                                                         jac_per_pol[:, d2, 1, :],
                                                         phi_sec_ders[..., d1 + d2]
                                                         ).numpy()

                phi_sec_ders_mapped[..., 2] += tf.einsum("pv,pv,ps->psv",
                                                         jac_per_pol[:, d1, 1, :],
                                                         jac_per_pol[:, d2, 1, :],
                                                         phi_sec_ders[..., d1 + d2]
                                                         ).numpy()

        return tf.convert_to_tensor(phi_sec_ders_mapped, dtype=tf.float64)

    def compute_adfs_and_grads(self, xy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xy)
            phi = self.compute_adfs(xy)

        phi_grad = tape.gradient(phi, xy)

        del tape

        return phi, phi_grad

    def compute_adfs_and_grads_and_second_derivatives(self, xy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xy)
            phi, phi_grad = self.compute_adfs_and_grads(xy)
            phi_x, phi_y = [phi_grad[:, :, :1], phi_grad[:, :, 1:]]

        phi_x_ders = tape.gradient(phi_x, xy)
        phi_y_ders = tape.gradient(phi_y, xy)
        phi_sec_ders = tf.concat([phi_x_ders, phi_y_ders[:, :, 1:]], -1)

        del tape

        return phi, phi_grad, phi_sec_ders

    def inside_polygon(self, xy: tf.Tensor) -> tf.Tensor:
        inside = 0
        for p in range(self.vertices.shape[0]):
            polygon = self.vertices[p, :, :]
            path = plt_path.Path(polygon)
            curr_inside = path.contains_points(xy[p, :, :])
            if p == 0:
                inside = tf.expand_dims(curr_inside, 0)
            else:
                curr_inside = tf.expand_dims(curr_inside, 0)
                inside = tf.concat([inside, curr_inside], 0)
        return inside

    def draw_adf(self, xy: tf.Tensor, adf: tf.Tensor, pol_id: int = 0, contour_plot: bool = True,
                 scatter_plot: bool = True) -> None:

        phi = tf.squeeze(adf[pol_id, :, :])
        inside = self.inside_polygon(xy)[pol_id, :]

        x = tf.boolean_mask(xy[pol_id, :, 0], inside)
        y = tf.boolean_mask(xy[pol_id, :, 1], inside)
        z = tf.boolean_mask(phi, inside)

        if contour_plot:
            plt.tricontourf(x, y, z, levels=20, cmap="viridis")
            cbar = plt.colorbar()

            segments = tf.concat([self.vertices, self.next_vertices], axis=2)
            for e in segments[pol_id, :, :]:
                plt.plot([e[0], e[2]], [e[1], e[3]], 'red', linewidth=2)
            plt.scatter(self.vertices[pol_id, :, 0], self.vertices[pol_id, :, 1], c='red', marker='o', s=40)
            cbar.ax.tick_params(labelsize=17)
            plt.axis("equal")
            plt.title("ADF")
            plt.show()

        if scatter_plot:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(projection='3d')

            ax.scatter(x, y, z)
            plt.title("ADF")
            plt.show()

    def draw_g(self, xy: tf.Tensor, g: tf.Tensor, pol_id: int = 0, dof_id: int = 0, contour_plot: bool = True,
               scatter_plot: bool = True) -> None:

        z = g[pol_id, :, dof_id]
        inside = self.inside_polygon(xy)[pol_id, :]

        x = tf.boolean_mask(xy[pol_id, :, 0], inside)
        y = tf.boolean_mask(xy[pol_id, :, 1], inside)
        z = tf.boolean_mask(z, inside)

        if contour_plot:
            plt.tricontourf(x, y, z, levels=20, cmap="viridis")
            cbar = plt.colorbar()

            segments = tf.concat([self.vertices, self.next_vertices], axis=2)
            for e in segments[pol_id, :, :]:
                plt.plot([e[0], e[2]], [e[1], e[3]], 'red', linewidth=2)
            plt.scatter(self.vertices[pol_id, :, 0], self.vertices[pol_id, :, 1], c='red', marker='o', s=40)
            cbar.ax.tick_params(labelsize=17)
            plt.title("g")
            plt.axis("equal")
            plt.show()

        if scatter_plot:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(projection='3d')

            ax.scatter(x, y, z)
            plt.title("g")
            plt.show()

    def draw_g_on_one_edge(self, xy: tf.Tensor, pol_id: int = 0, edge_id: int = 0):

        dist = None
        match self.boundary_method_type_g:
            case BoundaryMethodType.line:
                dist = self.compute_phi_k_line(xy)
            case BoundaryMethodType.segment:
                phi_k_line = self.compute_phi_k_line(xy)
                dist = self.compute_phi_k_segment(xy, phi_k_line)
            case _:
                raise ValueError("Unknown boundary method type.")

        dist = dist[pol_id, :, edge_id]

        plt.tricontourf(xy[pol_id, :, 0], xy[pol_id, :, 1], dist, levels=20, cmap="viridis")
        cbar = plt.colorbar()

        segments = tf.concat([self.vertices, self.next_vertices], axis=2)
        for e in segments[pol_id, :, :]:
            plt.plot([e[0], e[2]], [e[1], e[3]], 'blue', linewidth=1)
        plt.plot([segments[pol_id, edge_id, 0], segments[pol_id, edge_id, 2]],
                 [segments[pol_id, edge_id, 1], segments[pol_id, edge_id, 3]], 'red', linewidth=3)

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
