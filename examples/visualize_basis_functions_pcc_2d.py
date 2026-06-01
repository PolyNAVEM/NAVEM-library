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
import tensorflow as tf
import os
from pypolydim.export_vtk_utilities import ExportVTKUtilities
from NAVEM.PCC_2D import LocalSpace_PCC_2D
from NAVEM.geometry.mesh_utilities import compute_geometric_properties_mesh_2
import argparse
from pypolydim import gedim, polydim
from Elliptic_PCC_2D.program_utilities import create_mesh
from NAVEM.PCC_2D import NAVEM_PCC_2D
from numpy.typing import NDArray
from typing import Dict
from pathlib import Path


def main():

    program_folder = str(Path(__file__).resolve().parent)

    # Reading from command line all required information
    parser = argparse.ArgumentParser()
    parser.add_argument('-order', '--method-order', dest='method_order', default=1, type=int, help="Method order")
    parser.add_argument('-method', '--method-type', dest='method_type', default=1, type=int,
                        help="Method type: 1 - NAVEM; 2 - FEM; 3 - VEM")
    parser.add_argument('-mesh', '--mesh-type', dest='mesh_type', default=5, type=int, help="Mesh type")
    parser.add_argument('-tol1', '--tolerance-1-d', dest='tolerance1_d', default=1.0e-12, type=float, help="Geometric Tolerance 1D")
    parser.add_argument('-tol2', '--tolerance-2-d', dest='tolerance2_d', default=1.0e-14, type=float, help="Geometric Tolerance 2D")
    parser.add_argument('-area', '--mesh-max-relative-area', dest='max_relative_area', default=0.1, type=float, help="Mesh max relative area")
    parser.add_argument('-export', '--export-path', dest='export_path', default=program_folder + '/../Export/Elliptic_PCC_2D', type=str, help="Export Path")
    parser.add_argument('-import', '--import-path', dest='import_path', default='./', type=str, help="Mesh Import Path")
    parser.add_argument('-df', '--dictionary-file', dest='dictionary_file', default=program_folder + '/../TrainedModels/B-NAVEM', type=str, help="Dictionary file")
    args = parser.parse_args()

    tf.keras.backend.set_floatx('float64')
    tf.keras.backend.set_epsilon(args.tolerance1_d)

    export_file_path = args.export_path
    if not os.path.exists(export_file_path):
        os.makedirs(export_file_path)

    # Mesh file path
    export_basis_path = args.export_path + "/BasisFunctions"
    if not os.path.exists(export_basis_path):
        os.makedirs(export_basis_path)

    export_polygon_path = args.export_path + "/Polygons"
    if not os.path.exists(export_polygon_path):
        os.makedirs(export_polygon_path)

    mesh_type = polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D(args.mesh_type)
    method_type = LocalSpace_PCC_2D.MethodTypes(args.method_type)
    method_order = args.method_order

    print("Create mesh...")
    pde_domain = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D()

    pde_domain.vertices = np.array([[0.0, 1.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 0.0]])

    pde_domain.area = 1.0
    pde_domain.shape_type = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D.Domain_Shape_Types.parallelogram

    geometry_utilities_config = gedim.GeometryUtilitiesConfig()
    geometry_utilities_config.tolerance1_d = args.tolerance1_d
    geometry_utilities_config.tolerance2_d = args.tolerance2_d
    geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
    mesh_utilities = gedim.MeshUtilities()
    vtk_utilities = ExportVTKUtilities()

    mesh_data = gedim.MeshMatrices()
    mesh = gedim.MeshMatricesDAO(mesh_data)

    create_mesh(geometry_utilities, mesh_utilities, mesh_type, args.max_relative_area, args.import_path, pde_domain, mesh)

    print("Compute Geometric Properties...")
    mesh_geometric_data = compute_geometric_properties_mesh_2(geometry_utilities, mesh_utilities, mesh)

    reference_element_data = LocalSpace_PCC_2D.ReferenceElementData(method_type, method_order)
    reference_element_data.navem_categories = NAVEM_PCC_2D.categorize_elements_by_vertex_number(method_order, mesh, mesh_geometric_data, args.dictionary_file)

    print("Compute Evaluation Points...")
    num_cell_2 = mesh.cell2_d_total_number()

    evaluation_points: Dict[int, NDArray[np.float64]] = {}

    quadrature = polydim.vem.quadrature.VEM_Quadrature_2D()
    reference_quadrature = gedim.quadrature.Quadrature_Gauss2D_Triangle()
    reference_quadrature_data = reference_quadrature.fill_points_and_weights(20)

    points_on_each_edge = 31
    pts_on_01 = np.linspace(0.0, 1.0, points_on_each_edge)
    for c in range(num_cell_2):

        vertices = mesh_geometric_data.mesh_geometric_data.cell2_ds_vertices[c]
        n_edges = vertices.shape[1]
        boundary_points = np.zeros([3, points_on_each_edge * n_edges])

        offset = 0
        for i in range(n_edges):
            origin = vertices[:, i:(i + 1)]
            end = vertices[:, ((i + 1) % n_edges):((i + 1) % n_edges) + 1]
            for s in range(points_on_each_edge):
                boundary_points[:, offset:offset + 1] = origin + pts_on_01[s] * (end - origin)
                offset += 1

        internal_quadrature = quadrature.polygon_internal_quadrature(reference_quadrature_data,
                                                                     mesh_geometric_data.mesh_geometric_data.cell2_ds_triangulations[c])

        evaluation_points[c] = np.concatenate([boundary_points, internal_quadrature.points], axis=1)

    print("Export Polygons and Basis Functions...")
    local_space_data = LocalSpace_PCC_2D.LocalSpaceData(geometry_utilities, mesh_geometric_data, reference_element_data)
    evaluation_navem_input_output = None
    match method_type:
        case LocalSpace_PCC_2D.MethodTypes.NAVEM:
            evaluation_navem_input_output = NAVEM_PCC_2D.create_navem_input_output(geometry_utilities,
                                                                                   mesh_geometric_data,
                                                                                   reference_element_data.navem_categories,
                                                                                   evaluation_points)
        case _:
            pass

    for c in range(num_cell_2):

        local_space_data.create_local_space(geometry_utilities_config.tolerance1_d,
                                            geometry_utilities_config.tolerance2_d,
                                            mesh_geometric_data,
                                            c,
                                            reference_element_data)

        basis_functions_values = local_space_data.basis_functions_values(reference_element_data,
                                                                         polydim.vem.pcc.ProjectionTypes.pi0k,
                                                                         evaluation_points[c],
                                                                         evaluation_navem_input_output)

        num_do_fs = basis_functions_values.shape[1]
        point_data: Dict[str, NDArray[np.float64]] = {}
        for i in range(num_do_fs):
            point_data["Basis" + str(i)] = basis_functions_values[:, i]

        vtk_utilities.export_points(export_basis_path + "/polygons_" + str(c) + ".vtu", evaluation_points[c], point_data=point_data)
        vtk_utilities.export_polygons(export_polygon_path + "/polygons_" + str(c) + ".vtu", mesh.cell0_ds_coordinates(), [mesh.cell2_d_vertices(c)])


if __name__ == '__main__':
    main()
