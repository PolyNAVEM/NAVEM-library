import numpy as np
import matplotlib.pyplot as plt
import random
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
from pypolydim import gedim
from typing import Tuple
from NAVEM.geometry.geometry_utilities import polygon_is_simple, compute_polygon_kernel, compute_polygon_interior_angles
from numpy.typing import NDArray

class PolygonGenerator:

    def __init__(self, geometry_utilities: gedim.GeometryUtilities):
        self.geometry_utilities = geometry_utilities

    @staticmethod
    def plot_polygon(polygon: NDArray[np.float64], kernel: NDArray[np.float64] = None, internal_point: NDArray[np.float64] = None) -> None:
        plt.figure()
        plt.gca().set_aspect("equal")

        vertices = polygon
        num_vertices = vertices.shape[1]
        coord = np.zeros([3, num_vertices + 1])
        coord[:, 0:num_vertices] = vertices
        coord[:, num_vertices] = vertices[:, 0]
        plt.plot(coord[0, :], coord[1, :], label='polygon')

        for i in range(num_vertices):
            plt.text(float(vertices[0, i]), float(vertices[1, i]), str(i), horizontalalignment="center", verticalalignment="center")

        if kernel is not None:
            vertices = kernel
            num_vertices = vertices.shape[1]
            coord = np.zeros([3, num_vertices + 1])
            coord[:, 0:num_vertices] = vertices
            coord[:, num_vertices] = vertices[:, 0]
            plt.plot(coord[0, :], coord[1, :], label='kernel')

        if internal_point is not None:
            plt.plot(internal_point[0], internal_point[1], "bo", linewidth=0.4, label='internal point')

        plt.legend()
        plt.show()


    def generate_simple_random_polygon(self, num_polygons: int, num_vertices: int, plot_generate_elements: bool = False) -> Tuple[gedim.MeshMatrices, gedim.MeshMatricesDAO]:

        mesh_data = gedim.MeshMatrices()
        mesh = gedim.MeshMatricesDAO(mesh_data)
        mesh.initialize_dimension(2)

        random.seed(5)

        count_vertex = 0
        count_el = 0

        while count_el < num_polygons:

            polygon = random_polygon(num_points=num_vertices)

            vertices = np.zeros([3, num_vertices])

            for i, (x, y) in enumerate(polygon):
                vertices[0, i] = x
                vertices[1, i] = y

            if vertices[0, 0] is None or np.isnan(vertices[0, 0]):
                continue


            if not polygon_is_simple(self.geometry_utilities, vertices):
                continue

            if plot_generate_elements:
                self.plot_polygon(vertices)

            id_new_vertices = mesh.cell0_d_append(num_vertices)
            for v in range(num_vertices):

                mesh.cell0_d_insert_coordinates(id_new_vertices, vertices[:, v])
                mesh.cell0_d_set_marker(id_new_vertices, 0)
                mesh.cell0_d_set_state(id_new_vertices, True)

                id_new_vertices += 1
                count_vertex += 1

            id_new_edges = mesh.cell1_d_append(num_vertices)

            vertices_edges = []
            [vertices_edges.append(count_vertex - (num_vertices - i)) for i in range(num_vertices)]

            for e in range(num_vertices):
                mesh.cell1_d_insert_extremes(id_new_edges, vertices_edges[e], vertices_edges[(e + 1) % num_vertices])
                mesh.cell1_d_set_marker(id_new_edges, 0)
                mesh.cell1_d_set_state(id_new_edges, True)

                id_new_edges += 1

            id_new_cell = mesh.cell2_d_append(1)

            mesh.cell2_d_initialize_vertices(id_new_cell, num_vertices)
            mesh.cell2_d_initialize_edges(id_new_cell, num_vertices)
            for v in range(num_vertices):
                mesh.cell2_d_insert_vertex(id_new_cell, v, vertices_edges[v])
                mesh.cell2_d_insert_edge(id_new_cell, v, vertices_edges[v])

            mesh.cell2_d_set_marker(id_new_cell, 0)
            mesh.cell2_d_set_state(id_new_cell, True)

            count_el += 1

        return mesh_data, mesh

    def generate_star_shaped_random_polygon(self, num_polygons: int, num_vertices: int, plot_generate_elements: bool = False) -> Tuple[gedim.MeshMatrices, gedim.MeshMatricesDAO]:

        mesh_data = gedim.MeshMatrices()
        mesh = gedim.MeshMatricesDAO(mesh_data)
        mesh.initialize_dimension(2)

        random.seed(5)

        count_vertex = 0
        count_el = 0

        while count_el < num_polygons:

            polygon = random_polygon(num_points=num_vertices)

            vertices = np.zeros([3, num_vertices])

            for i, (x, y) in enumerate(polygon):
                vertices[0, i] = x
                vertices[1, i] = y

            if vertices[0, 0] is None or np.isnan(vertices[0, 0]):
                continue

            if not polygon_is_simple(self.geometry_utilities, vertices):
                continue

            internal_angles = compute_polygon_interior_angles(vertices)
            kernel = compute_polygon_kernel(self.geometry_utilities, vertices, internal_angles)

            if kernel.shape[1] == 0:
                continue

            kernel_area = self.geometry_utilities.polygon_area(kernel)
            polygon_area = self.geometry_utilities.polygon_area(vertices)

            if kernel_area < self.geometry_utilities.tolerance2_d() or kernel_area < polygon_area * 0.2:
                continue

            if plot_generate_elements:
                self.plot_polygon(vertices, kernel)

            id_new_vertices = mesh.cell0_d_append(num_vertices)
            for v in range(num_vertices):
                mesh.cell0_d_insert_coordinates(id_new_vertices, vertices[:, v])
                mesh.cell0_d_set_marker(id_new_vertices, 0)
                mesh.cell0_d_set_state(id_new_vertices, True)

                id_new_vertices += 1
                count_vertex += 1

            id_new_edges = mesh.cell1_d_append(num_vertices)

            vertices_edges = []
            [vertices_edges.append(count_vertex - (num_vertices - i)) for i in range(num_vertices)]

            for e in range(num_vertices):
                mesh.cell1_d_insert_extremes(id_new_edges, vertices_edges[e], vertices_edges[(e + 1) % num_vertices])
                mesh.cell1_d_set_marker(id_new_edges, 0)
                mesh.cell1_d_set_state(id_new_edges, True)

                id_new_edges += 1

            id_new_cell = mesh.cell2_d_append(1)

            mesh.cell2_d_initialize_vertices(id_new_cell, num_vertices)
            mesh.cell2_d_initialize_edges(id_new_cell, num_vertices)
            for v in range(num_vertices):
                mesh.cell2_d_insert_vertex(id_new_cell, v, vertices_edges[v])
                mesh.cell2_d_insert_edge(id_new_cell, v, vertices_edges[v])

            mesh.cell2_d_set_marker(id_new_cell, 0)
            mesh.cell2_d_set_state(id_new_cell, True)

            count_el += 1

        return mesh_data, mesh


    def generate_random_concave_polygon(self, num_polygons: int, num_reflex_angles: int,
                                        num_vertices: int, tolerance_hanging_nodes: float,
                                        plot_generate_elements: bool = False) -> Tuple[gedim.MeshMatrices, gedim.MeshMatricesDAO]:

        mesh_data = gedim.MeshMatrices()
        mesh = gedim.MeshMatricesDAO(mesh_data)
        mesh.initialize_dimension(2)

        random.seed(5)

        count_vertex = 0
        count_el = 0

        while count_el < num_polygons:
            polygon = random_star_shaped_polygon(num_points=num_vertices)

            vertices = np.zeros([3, num_vertices])

            for i, (x, y) in enumerate(polygon):
                vertices[0, i] = x
                vertices[1, i] = y

            if vertices[0, 0] is None or np.isnan(vertices[0, 0]):
                continue

            if not polygon_is_simple(self.geometry_utilities, vertices):
                continue

            internal_angles = compute_polygon_interior_angles(vertices)
            kernel = compute_polygon_kernel(self.geometry_utilities, vertices, internal_angles)

            if kernel.shape[1] == 0:
                continue

            kernel_area = self.geometry_utilities.polygon_area(kernel)
            polygon_area = self.geometry_utilities.polygon_area(vertices)

            if kernel_area < self.geometry_utilities.tolerance2_d() or kernel_area < polygon_area * 0.2:
                continue

            is_concave = any(a > np.pi * (1.0 + tolerance_hanging_nodes) for a in internal_angles)

            if not is_concave:
                continue

            count_vertex_reflex = 0
            for v in range(num_vertices):
                if internal_angles[v] > np.pi * (1 + tolerance_hanging_nodes):
                    count_vertex_reflex += 1

            if count_vertex_reflex != num_reflex_angles:
                continue

            if plot_generate_elements:
                self.plot_polygon(vertices, kernel)

            id_new_vertices = mesh.cell0_d_append(num_vertices)
            for v in range(num_vertices):
                mesh.cell0_d_insert_coordinates(id_new_vertices, vertices[:, v])
                mesh.cell0_d_set_marker(id_new_vertices, 0)
                mesh.cell0_d_set_state(id_new_vertices, True)

                id_new_vertices += 1
                count_vertex += 1

            id_new_edges = mesh.cell1_d_append(num_vertices)

            vertices_edges = []
            [vertices_edges.append(count_vertex - (num_vertices - i)) for i in range(num_vertices)]

            for e in range(num_vertices):
                mesh.cell1_d_insert_extremes(id_new_edges, vertices_edges[e], vertices_edges[(e + 1) % num_vertices])
                mesh.cell1_d_set_marker(id_new_edges, 0)
                mesh.cell1_d_set_state(id_new_edges, True)

                id_new_edges += 1

            id_new_cell = mesh.cell2_d_append(1)

            mesh.cell2_d_initialize_vertices(id_new_cell, num_vertices)
            mesh.cell2_d_initialize_edges(id_new_cell, num_vertices)
            for v in range(num_vertices):
                mesh.cell2_d_insert_vertex(id_new_cell, v, vertices_edges[v])
                mesh.cell2_d_insert_edge(id_new_cell, v, vertices_edges[v])

            mesh.cell2_d_set_marker(id_new_cell, 0)
            mesh.cell2_d_set_state(id_new_cell, True)

            count_el += 1

        return mesh_data, mesh


    def generate_random_convex_polygon(self, num_polygons: int, num_vertices: int,
                                       tolerance_hanging_nodes: float,
                                       plot_generate_elements: bool = False) \
            -> Tuple[gedim.MeshMatrices, gedim.MeshMatricesDAO]:

        mesh_data = gedim.MeshMatrices()
        mesh = gedim.MeshMatricesDAO(mesh_data)
        mesh.initialize_dimension(2)

        random.seed(5)

        count_vertex = 0
        count_el = 0

        while count_el < num_polygons:

            polygon = random_convex_polygon(num_points=num_vertices)

            vertices = np.zeros([3, num_vertices])
            for i, (x, y) in enumerate(polygon):
                vertices[0, i] = x
                vertices[1, i] = y

            if vertices[0, 0] is None or np.isnan(vertices[0, 0]):
                continue

            if not polygon_is_simple(self.geometry_utilities, vertices):
                continue

            internal_angles = compute_polygon_interior_angles(vertices)
            is_concave = any(a > np.pi * (1.0 + tolerance_hanging_nodes) for a in internal_angles)

            if is_concave:
                continue

            if plot_generate_elements:
                self.plot_polygon(vertices)

            id_new_vertices = mesh.cell0_d_append(num_vertices)
            for v in range(num_vertices):
                mesh.cell0_d_insert_coordinates(id_new_vertices, vertices[:, v])
                mesh.cell0_d_set_marker(id_new_vertices, 0)
                mesh.cell0_d_set_state(id_new_vertices, True)

                id_new_vertices += 1
                count_vertex += 1

            id_new_edges = mesh.cell1_d_append(num_vertices)

            vertices_edges = []
            [vertices_edges.append(count_vertex - (num_vertices - i)) for i in range(num_vertices)]

            for e in range(num_vertices):
                mesh.cell1_d_insert_extremes(id_new_edges, vertices_edges[e], vertices_edges[(e + 1) % num_vertices])
                mesh.cell1_d_set_marker(id_new_edges, 0)
                mesh.cell1_d_set_state(id_new_edges, True)

                id_new_edges += 1

            id_new_cell = mesh.cell2_d_append(1)

            mesh.cell2_d_initialize_vertices(id_new_cell, num_vertices)
            mesh.cell2_d_initialize_edges(id_new_cell, num_vertices)
            for v in range(num_vertices):
                mesh.cell2_d_insert_vertex(id_new_cell, v, vertices_edges[v])
                mesh.cell2_d_insert_edge(id_new_cell, v, vertices_edges[v])

            mesh.cell2_d_set_marker(id_new_cell, 0)
            mesh.cell2_d_set_state(id_new_cell, True)

            count_el += 1

        return mesh_data, mesh
