from enum import Enum
from pypolydim import gedim
from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from src.NAVEM.NeuralNetwork import navem_network
from src.NAVEM.Utilities import NAVEMGenerators
import tensorflow as tf
from src.GeDiM.geometry.geometry_utilities import MeshGeometricData2D


class NAVEMType(Enum):
    NAVEM = 1
    B_NAVEM = 2
    P_NAVEM = 3

class Input:

    def __init__(self):
        self.points: NDArray[np.float64] = np.zeros((3,0))

class Output:

    def __init__(self):
        self.basis_values: NDArray[np.float64] = np.zeros((0,0))
        self.basis_derivatives_values: NDArray[np.float64] = np.zeros((2, 0, 0))

class NNDictionary:

    def __init__(self):
        self.list_elements: List[int] = []
        self.method_type: NAVEMType = NAVEMType.NAVEM
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

            method_type = NAVEMType(raw["method_type"])

            match method_type:
                case NAVEMType.NAVEM:
                    flags = navem_network.load_flags_from_dictionary(name_storage, raw)
                    categories[num_vertices].neural_network = navem_network.NAVEMNetwork(flags)
                    categories[num_vertices].neural_network.load_weights(name_storage + "/nn_weights.weights.h5").expect_partial()
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
                                               neural_network: navem_network.NAVEMNetwork,
                                               list_elements: Dict[int, Input]) -> Dict[int, Output]:


    navem_generators = NAVEMGenerators.NAVEMGenerators(geometry_utilities,
                                                       neural_network.flags["num_vertices"],
                                                       neural_network.flags["harmonic_degree"],
                                                       neural_network.flags["use_hanging_function"],
                                                       neural_network.flags["normalization_diameter"])

    num_vertices = neural_network.flags["num_vertices"]
    network_input_dimension = 2 + 2 * (num_vertices - 1)

    inputs = np.array([], dtype=np.float64).reshape(0, network_input_dimension - 2)
    num_generators = neural_network.flags["num_generators"]

    n_elements = len(list_elements)
    global_jac_inv = np.zeros([n_elements * num_vertices, 2, 2])

    n_local_pts = 0
    for (c, value) in list_elements:
        n_local_pts = value.input.points.shape[1]
        break

    super_vander = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))
    super_vander_dx = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))
    super_vander_dy = np.zeros((n_elements * num_vertices, n_local_pts, num_generators))

    for (c, value) in list_elements:

        polygon = mesh_geometric_data.cell2_ds_polygon[c]
        mapped_angles = np.expand_dims(np.array(mesh_geometric_data.cell2_ds_mapped_polygon_internal_angles[c]), axis=1)

        internal_nodes = value.input.points

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

    derivatives_coefficients = neural_network.nn_pol_deriv.call(inputs)
    dx_values = neural_network.nn_pol_deriv.apply_vandermonde(derivatives_coefficients, super_vander_dx)
    dy_values = neural_network.nn_pol_deriv.apply_vandermonde(derivatives_coefficients, super_vander_dy)
    dx_values = tf.reshape(dx_values, [-1]).numpy()
    dy_values = tf.reshape(dy_values, [-1]).numpy()

    output_values: Dict[int, Output] = {}

    offset_point = 0
    offset_func = 0
    for c in range(n_elements):

        output_values[c] = Output()
        output_values[c].basis_derivatives_values = np.zeros([2, n_local_pts, n_elements * num_vertices])
        output_values[c].basis_values = np.zeros([n_local_pts, n_elements * num_vertices])

        for _ in range(num_vertices):
            matrix_jac_inv_transpose = global_jac_inv[offset_func, 0:2, 0:2].T
            output_values[c].basis_derivatives_values[0, :, :] \
                = (matrix_jac_inv_transpose[0, 0] * dx_values[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[0, 1] * dy_values[offset_point: offset_point + n_local_pts])
            output_values[c].basis_derivatives_values[0, :, :]  \
                = (matrix_jac_inv_transpose[1, 0] * dx_values[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[1, 1] * dy_values[offset_point: offset_point + n_local_pts])

            output_values[c].basis_values \
                = basis_values[offset_point:offset_point + n_local_pts]

            offset_func += 1
            offset_point += n_local_pts

    return output_values

def exact_bc_navem_predict_basis_values_and_derivatives(geometry_utilities: gedim.GeometryUtilities,
                                                        mesh_geometric_data: MeshGeometricData2D,
                                                        neural_network: navem_network.NAVEMNetwork,
                                                        list_elements: Dict[int, Input]) -> Dict[int, Output]:

    num_vertices = neural_network.flags["num_vertices"]
    network_input_dimension = 2 * num_vertices

    n_elements = len(list_elements)
    global_jac_inv = np.zeros([n_elements * num_vertices, 2, 2])

    n_local_pts = 0
    for (c, value) in list_elements:
        n_local_pts = value.input.points.shape[1]
        break

    offset_func = 0
    offset_points = 0

    c_pol = 0

    xy_per_pol = np.zeros(shape=(n_elements, n_local_pts, 2))
    vertices_per_pol = np.zeros(shape=(n_elements, 2, num_vertices))
    jac_per_pol = np.zeros(shape=(n_elements, 2, 2, num_vertices))
    inputs = np.zeros(shape=(n_elements * n_local_pts * num_vertices, network_input_dimension))

    for (c, value) in list_elements:

        polygon = mesh_geometric_data.cell2_ds_polygon[c]

        internal_nodes = value.input.points

        inertia_mapped_internal_nodes, jac_inv_inertia = polygon.map_inertia_inv(internal_nodes)
        xy_per_pol[c_pol, :, :] = inertia_mapped_internal_nodes[:2, :].T
        vertices_per_pol[c_pol, :, :] = polygon.mapped_vertices[:2, :]

        for v_id in range(num_vertices):
            # Create input
            rotated_vertices, rotated_mapped_points, jac_inv, jac, inv_rescaling_factor = polygon.map_fix_vertex_inv(
                polygon.mapped_vertices,
                inertia_mapped_internal_nodes,
                v_id)

            mapped_vertices = np.roll(rotated_vertices, axis=1, shift=-v_id)
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

    neural_network.setup_model_global_input(xy_per_pol, vertices_per_pol, jac_per_pol, 1,
                                           geometry_utilities)
    net_output = neural_network.get_u_and_du(inputs).numpy()

    u = net_output[:, 0]
    du_ref_dx = net_output[:, 1]
    du_ref_dy = net_output[:, 2]

    offset_point = 0
    offset_func = 0
    output_values: Dict[int, Output] = {}
    for c in range(n_elements):

        output_values[c] = Output()
        output_values[c].basis_derivatives_values = np.zeros([2, n_local_pts, n_elements * num_vertices])
        output_values[c].basis_values = np.zeros([n_local_pts, n_elements * num_vertices])

        for _ in range(num_vertices):
            matrix_jac_inv_transpose = global_jac_inv[offset_func, 0:2, 0:2].T
            output_values[c].basis_derivatives_values[0, :, :] \
                = (matrix_jac_inv_transpose[0, 0] * du_ref_dx[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[0, 1] * du_ref_dy[offset_point: offset_point + n_local_pts])
            output_values[c].basis_derivatives_values[1, :, :] \
                = (matrix_jac_inv_transpose[1, 0] * du_ref_dx[offset_point: offset_point + n_local_pts]
                   + matrix_jac_inv_transpose[1, 1] * du_ref_dy[offset_point: offset_point + n_local_pts])

            output_values[c].basis_values\
                = u[offset_point:offset_point + n_local_pts]

            offset_func += 1
            offset_point += n_local_pts

    return output_values

def reproduce_polynomials(polygon_vertices: NDArray[np.float64], basis_function_values: NDArray[np.float64], basis_function_derivatives_values: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:


    basis_function_values[:, -1] = 1.0 - np.sum(basis_function_values[:, 0:-1], axis=1)
    basis_function_derivatives_values[0, :, -1] = - np.sum(basis_function_derivatives_values[0, :, 0:-1],
                                                           axis=1)
    basis_function_derivatives_values[1, :, -1] = - np.sum(basis_function_derivatives_values[1, :, 0:-1],
                                                           axis=1)

    return basis_function_values, basis_function_derivatives_values