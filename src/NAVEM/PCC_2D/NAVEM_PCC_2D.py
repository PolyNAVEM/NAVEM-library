from enum import Enum
from pypolydim import gedim
from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from src.NAVEM.NeuralNetwork import h_navem_network, b_navem_network, p_navem_network
from src.NAVEM.NeuralNetwork import exact_bc_navem_network_utilities
from src.NAVEM.NeuralNetwork.exact_bc_navem_network_utilities import SetupDerivatives
from src.NAVEM.Utilities import NAVEMGenerators
import tensorflow as tf
from src.GeDiM.geometry.geometry_utilities import MeshGeometricData2D


class NAVEMType(Enum):
    H_NAVEM = 1
    B_NAVEM = 2
    P_NAVEM = 3

class Output:

    def __init__(self):
        self.basis_values: NDArray[np.float64] = np.zeros((0,0))
        self.basis_derivatives_values: List[NDArray[np.float64]] = [np.zeros((0,0)) for _ in range(2)]
        self.basis_laplacian_values: NDArray[np.float64] = np.zeros((0,0))

class InputOutput:

    internal_quadrature: gedim.quadrature.QuadratureData
    basis_values: NDArray[np.float64]
    basis_derivatives_values: List[NDArray[np.float64]]
    basis_laplacian_values: NDArray[np.float64]

    def __init__(self):
        pass

class NNDictionary:

    def __init__(self):
        self.list_elements: List[int] = []
        self.method_type: NAVEMType = NAVEMType.H_NAVEM
        self.neural_network = None

def categorize_elements_by_vertex_number(mesh: gedim.MeshMatricesDAO, dictionary_file_path: str) -> Dict[int, NNDictionary]:

    categories: Dict[int, NNDictionary] = {}
    with open(dictionary_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            num_vertices, name_storage = line.split("=", 1)
            num_vertices = int(num_vertices.strip())
            name_storage = name_storage.strip()

            categories[num_vertices] = NNDictionary()

            raw: Dict[str, str] = {}
            with open(name_storage + "/dictionary.txt", "r") as dictionary_file:
                for dictionary_line in dictionary_file:
                    dictionary_line = dictionary_line.strip()
                    if not dictionary_line:
                        continue

                    key, value = dictionary_line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    raw[key] = value

            categories[num_vertices].method_type = NAVEMType(int(raw["method_type"]))

            match categories[num_vertices].method_type:
                case NAVEMType.H_NAVEM:
                    flags = h_navem_network.load_flags_from_dictionary(name_storage, raw)
                    categories[num_vertices].neural_network = h_navem_network.HNAVEMNetworksContainer(flags)
                    categories[num_vertices].neural_network.load_weights(name_storage)
                case NAVEMType.B_NAVEM:
                    flags = exact_bc_navem_network_utilities.load_flags_from_dictionary(name_storage, raw)
                    categories[num_vertices].neural_network = b_navem_network.BNAVEMNetwork(flags)
                    categories[num_vertices].neural_network.load_model(name_storage)
                case NAVEMType.P_NAVEM:
                    flags = exact_bc_navem_network_utilities.load_flags_from_dictionary(name_storage, raw)
                    categories[num_vertices].neural_network = p_navem_network.PNAVEMNetwork(flags)
                    categories[num_vertices].neural_network.load_model(name_storage)
                case _:
                    raise ValueError("Unknown method type")

    for c in range(mesh.cell2_d_total_number()):

        num_vertices = mesh.cell2_d_number_vertices(c)

        if num_vertices == 3:
            continue

        if num_vertices not in categories:
            raise ValueError("Not valid element")

        categories[num_vertices].list_elements.append(c)

    return categories


def navem_predict_basis_values_and_derivatives(geometry_utilities: gedim.GeometryUtilities,
                                               mesh_geometric_data: MeshGeometricData2D,
                                               neural_network: h_navem_network.HNAVEMNetworksContainer,
                                               list_points: Dict[int, NDArray[np.float64]],
                                               predict_laplacian: bool = False) -> Dict[int, Output]:

    if not predict_laplacian:
        raise ValueError("not implemented method")

    navem_generators = NAVEMGenerators.NAVEMGenerators(geometry_utilities,
                                                       neural_network.flags["num_vertices"],
                                                       neural_network.flags["harmonic_degree"],
                                                       neural_network.flags["use_hanging_function"],
                                                       neural_network.flags["normalization_diameter"])

    num_vertices = neural_network.flags["num_vertices"]
    network_input_dimension = 2 + 2 * (num_vertices - 1)

    inputs = np.array([], dtype=np.float64).reshape(0, network_input_dimension - 2)
    num_generators = neural_network.flags["num_generators"]

    n_elements = len(list_points)
    global_jac_inv = np.zeros([n_elements * num_vertices, 2, 2])

    n_local_pts = 0
    for c, points in list_points.items():
        n_local_pts = points.shape[1]
        break

    super_vander = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))
    super_vander_dx = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))
    super_vander_dy = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))

    for c, internal_nodes in list_points.items():

        polygon = mesh_geometric_data.cell2_ds_polygon[c]
        mapped_angles = np.expand_dims(np.array(mesh_geometric_data.cell2_ds_mapped_polygon_internal_angles[c]), axis=1)


        for v_id in range(num_vertices):

            # Create input
            mapped_vertices, mapped_pts, jac_inv, scaling = polygon.map_f_inv(internal_nodes, v_id)
            mapped_vertices = np.roll(mapped_vertices, axis=1, shift=-v_id)
            internal_angles = np.roll(mapped_angles, shift=-v_id)
            vertex_distance = scaling * np.roll(polygon.mapped_max_vertex_distance, shift=-v_id)
            global_jac_inv[c * num_vertices + v_id, :, :] = jac_inv[:2, :2]

            c_inputs = np.expand_dims(mapped_vertices[:2, 1:].flatten(), 0)

            inputs = np.concatenate([inputs, c_inputs])

            # Store the vandermonde matrix that will be used
            local_vandermonde, local_vandermonde_grads = (
                navem_generators.vander_and_vander_derivatives(mapped_pts, mapped_vertices,
                                                               internal_angles=internal_angles,
                                                               vertex_distance=vertex_distance))

            super_vander[c * num_vertices + v_id, :, :] = local_vandermonde
            super_vander_dx[c * num_vertices + v_id, :, :] = local_vandermonde_grads[0, :, :]
            super_vander_dy[c * num_vertices + v_id, :, :] = local_vandermonde_grads[1, :, :]


    inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)

    super_vander = tf.convert_to_tensor(super_vander, dtype=tf.float64)
    super_vander_dx = tf.convert_to_tensor(super_vander_dx, dtype=tf.float64)
    super_vander_dy = tf.convert_to_tensor(super_vander_dy, dtype=tf.float64)

    basis_coefficients = neural_network.nn_basis_function.call(inputs)
    basis_values = neural_network.nn_basis_function.apply_vandermonde(basis_coefficients, super_vander)
    basis_values = tf.reshape(basis_values, [-1]).numpy()

    derivatives_coefficients = neural_network.nn_basis_derivatives.call(inputs)
    dx_values = neural_network.nn_basis_derivatives.apply_vandermonde(derivatives_coefficients, super_vander_dx)
    dy_values = neural_network.nn_basis_derivatives.apply_vandermonde(derivatives_coefficients, super_vander_dy)
    dx_values = tf.reshape(dx_values, [-1]).numpy()
    dy_values = tf.reshape(dy_values, [-1]).numpy()

    output_values: Dict[int, Output] = {}

    offset_point = 0
    offset_func = 0
    for c in range(n_elements):

        output_values[c] = Output()
        output_values[c].basis_derivatives_values[0] = np.zeros([n_local_pts, num_vertices])
        output_values[c].basis_derivatives_values[1] = np.zeros([n_local_pts, num_vertices])
        output_values[c].basis_values = np.zeros([n_local_pts, num_vertices])

        output_values[c].basis_laplacian_values = None

        for i in range(num_vertices):
            matrix_jac_inv_transpose = global_jac_inv[offset_func, 0:2, 0:2].T
            output_values[c].basis_derivatives_values[0][:, i] \
                = (matrix_jac_inv_transpose[0, 0] * dx_values[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[0, 1] * dy_values[offset_point: offset_point + n_local_pts])
            output_values[c].basis_derivatives_values[1][:, i]  \
                = (matrix_jac_inv_transpose[1, 0] * dx_values[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[1, 1] * dy_values[offset_point: offset_point + n_local_pts])

            output_values[c].basis_values[:, i] \
                = basis_values[offset_point:offset_point + n_local_pts]

            offset_func += 1
            offset_point += n_local_pts

    return output_values


def exact_bc_navem_predict_basis_values_and_derivatives(geometry_utilities: gedim.GeometryUtilities,
                                                        mesh_geometric_data: MeshGeometricData2D,
                                                        neural_network,
                                                        list_elements: Dict[int, NDArray[np.float64]],
                                                        predict_laplacian: bool = False) -> Dict[int, Output]:

    num_vertices = neural_network.flags["num_vertices"]
    network_input_dimension = 2 * num_vertices

    n_elements = len(list_elements)
    global_jac_inv = np.zeros([n_elements * num_vertices, 2, 2])

    n_local_pts = 0
    for c, points in list_elements.items():
        n_local_pts = points.shape[1]
        break

    offset_func = 0
    offset_points = 0

    c_pol = 0

    xy_per_pol = np.zeros(shape=(n_elements, n_local_pts, 2))
    vertices_per_pol = np.zeros(shape=(n_elements, 2, num_vertices))
    jac_per_pol = np.zeros(shape=(n_elements, 2, 2, num_vertices))
    inputs = np.zeros(shape=(n_elements * n_local_pts * num_vertices, network_input_dimension))

    for c, internal_nodes in list_elements.items():

        polygon = mesh_geometric_data.cell2_ds_polygon[c]

        inertia_mapped_internal_nodes, jac_inv_inertia = polygon.map_inertia_inv(internal_nodes)
        xy_per_pol[c_pol, :, :] = inertia_mapped_internal_nodes[:2, :].T
        vertices_per_pol[c_pol, :, :] = polygon.mapped_vertices[:2, :]

        for v_id in range(num_vertices):
            # Create input
            rotated_vertices, rotated_mapped_points, jac_inv, jac, _ = polygon.map_fix_vertex_inv(
                polygon.mapped_vertices,
                inertia_mapped_internal_nodes,
                v_id)

            mapped_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)

            if predict_laplacian:
                global_jac_inv[offset_func, :, :] = np.eye(2)
            else:
                global_jac_inv[offset_func, :, :] = jac_inv[:2, :2] @ jac_inv_inertia[:2, :2]

            flat_rotated_vertices = mapped_vertices[:2, 1:].flatten()
            rep_flat_rot_vertices = np.tile(flat_rotated_vertices[np.newaxis, :], [n_local_pts, 1])
            c_inputs = np.concatenate([rotated_mapped_points[:2, :].T, rep_flat_rot_vertices], axis=1)

            inputs[offset_func * n_local_pts: (offset_func + 1) * n_local_pts, :] = c_inputs
            jac_per_pol[c_pol, :, :, v_id] = jac[:2, :2]
            offset_func += 1

        c_pol += 1
        offset_points += 1

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)

    du_ref_dxx = None
    du_ref_dxy = None
    du_ref_dyy = None
    if predict_laplacian:
        neural_network.setup_model_global_input(xy_per_pol, vertices_per_pol, jac_per_pol, SetupDerivatives.basis_and_derivatives_and_laplacian, geometry_utilities)
        laplacian_output = neural_network.get_second_derivatives_u(inputs).numpy()
        du_ref_dxx = laplacian_output[:, 0]
        du_ref_dxy = laplacian_output[:, 1]
        du_ref_dyy = laplacian_output[:, 2]
    else:
        neural_network.setup_model_global_input(xy_per_pol, vertices_per_pol, jac_per_pol, SetupDerivatives.basis_and_derivatives, geometry_utilities)

    net_output = neural_network.get_u_and_du(inputs).numpy()

    u = net_output[:, 0]
    du_ref_dx = net_output[:, 1]
    du_ref_dy = net_output[:, 2]

    offset_point = 0
    offset_func = 0
    output_values: Dict[int, Output] = {}
    for c in range(n_elements):

        output_values[c] = Output()
        output_values[c].basis_derivatives_values[0] = np.zeros([n_local_pts, num_vertices])
        output_values[c].basis_derivatives_values[1] = np.zeros([n_local_pts, num_vertices])
        output_values[c].basis_values = np.zeros([n_local_pts, num_vertices])

        if predict_laplacian:
            output_values[c].basis_laplacian_values = np.zeros([n_local_pts, num_vertices])
        else:
            output_values[c].basis_laplacian_values = None

        for i in range(num_vertices):
            matrix_jac_inv_transpose = global_jac_inv[offset_func, 0:2, 0:2].T
            output_values[c].basis_derivatives_values[0][:, i] \
                = (matrix_jac_inv_transpose[0, 0] * du_ref_dx[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[0, 1] * du_ref_dy[offset_point: offset_point + n_local_pts])
            output_values[c].basis_derivatives_values[1][:, i] \
                = (matrix_jac_inv_transpose[1, 0] * du_ref_dx[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[1, 1] * du_ref_dy[offset_point: offset_point + n_local_pts])

            output_values[c].basis_values[:, i]\
                = u[offset_point:offset_point + n_local_pts]

            if predict_laplacian:
                b_lap = matrix_jac_inv_transpose.T @ matrix_jac_inv_transpose
                output_values[c].basis_laplacian_values[:, i] = (b_lap[0, 0] * du_ref_dxx[offset_point: offset_point + n_local_pts]
                                                                 + b_lap[0, 1] * du_ref_dxy[offset_point: offset_point + n_local_pts]
                                                                 + b_lap[1, 0] * du_ref_dxy[offset_point: offset_point + n_local_pts]
                                                                 + b_lap[1, 1] * du_ref_dyy[offset_point: offset_point + n_local_pts])

            offset_func += 1
            offset_point += n_local_pts

    return output_values

def reproduce_polynomials(polygon_vertices: NDArray[np.float64],
                          basis_function_values: NDArray[np.float64],
                          basis_function_derivatives_values: List[NDArray[np.float64]],
                          basis_laplacian_values: NDArray[np.float64] = None) \
        -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], NDArray[np.float64]]:

    basis_function_values[:, -1] = 1.0 - np.sum(basis_function_values[:, 0:-1], axis=1)
    basis_function_derivatives_values[0][:, -1] = - np.sum(basis_function_derivatives_values[0][:, 0:-1],
                                                           axis=1)
    basis_function_derivatives_values[1][:, -1] = - np.sum(basis_function_derivatives_values[1][:, 0:-1],
                                                           axis=1)

    if basis_laplacian_values is not None:
        basis_laplacian_values[:, -1] = - np.sum(basis_laplacian_values[:, 0:-1], axis=1)

    return basis_function_values, basis_function_derivatives_values, basis_laplacian_values

def create_navem_input_output(geometry_utilities: gedim.GeometryUtilities,
                              mesh_geometric_data: MeshGeometricData2D,
                              navem_categories: Dict[int, NNDictionary],
                              evaluation_points: Dict[int, NDArray[np.float64]],
                              evaluation_weights: Dict[int, NDArray[np.float64]] = None,
                              predict_laplacian: bool = False) -> Dict[int, InputOutput]:


        navem_input_output: Dict[int, InputOutput] = {}

        for num_vertices, dictionary in navem_categories.items():

            if len(dictionary.list_elements) == 0:
                continue

            list_points: Dict[int, NDArray[np.float64]] = {}

            for c in dictionary.list_elements:

                if c not in evaluation_points:
                    continue

                navem_input_output[c] = InputOutput()
                navem_input_output[c].internal_quadrature = gedim.quadrature.QuadratureData()
                navem_input_output[c].internal_quadrature.points = evaluation_points[c]
                navem_input_output[c].internal_quadrature.weights = evaluation_weights[c] if evaluation_weights is not None else np.zeros([0])

                list_points[c] = navem_input_output[c].internal_quadrature.points

            outputs: Dict[int, Output] = {}

            match dictionary.method_type:
                case NAVEMType.H_NAVEM:
                    outputs \
                        = navem_predict_basis_values_and_derivatives(geometry_utilities,
                                                                     mesh_geometric_data,
                                                                     dictionary.neural_network,
                                                                     list_points,
                                                                     predict_laplacian)
                case NAVEMType.B_NAVEM | NAVEMType.P_NAVEM:
                    outputs \
                        = exact_bc_navem_predict_basis_values_and_derivatives(geometry_utilities,
                                                                              mesh_geometric_data,
                                                                              dictionary.neural_network,
                                                                              list_points,
                                                                              predict_laplacian)
                case _:
                    raise ValueError("Not valid method type")

            for c in dictionary.list_elements:
                navem_input_output[c].basis_values, navem_input_output[c].basis_derivatives_values, navem_input_output[c].basis_laplacian_values \
                    = reproduce_polynomials(
                    mesh_geometric_data.mesh_geometric_data.cell2_ds_vertices[c],
                    outputs[c].basis_values,
                    outputs[c].basis_derivatives_values,
                    outputs[c].basis_laplacian_values)

        return navem_input_output
