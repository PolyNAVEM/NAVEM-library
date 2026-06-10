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

import os
import csv
import numpy as np
import shutil
from pathlib import Path


def import_errors_pcc(export_path, method_type, method_order, test_type):
    errors_file = os.path.join(export_path,
                               "Errors_" + str(test_type) + "_" + str(method_type) + "_" + str(method_order) + ".csv")
    errors = []
    with open(errors_file, newline='') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=';')
        data = list(file_reader)

        counter = 0
        for row in data:
            errors_row = []
            if counter == 0:
                errors_row.append(row[7])
                errors_row.append(row[8])
                errors_row.append(row[9])
                errors_row.append(row[10])
                errors_row.append(row[11])
            else:
                errors_row.append(float(row[7]))
                errors_row.append(float(row[8]))
                errors_row.append(float(row[9]))
                errors_row.append(float(row[10]))
                errors_row.append(float(row[11]))
            errors.append(errors_row)
            counter += 1

    return errors


def test_errors_pcc(errors, method_order, tol):
    num_rows = len(errors)

    if num_rows == 2:
        print('\x1b[0;31;40m' + "Num. Ref. 1: ", abs(errors[1][1]) / abs(errors[1][3]),
              abs(errors[1][2]) / abs(errors[1][4]), '\x1b[0m')
        assert abs(errors[1][1]) < tol * abs(errors[1][3])
        assert abs(errors[1][2]) < tol * abs(errors[1][4])
    else:
        errors = np.array(errors[1:])
        slope_l2 = float(np.polyfit(np.log(errors[:, 0]), np.log(errors[:, 1]), 1)[0])
        slope_h1 = float(np.polyfit(np.log(errors[:, 0]), np.log(errors[:, 2]), 1)[0])
        print('\x1b[0;31;40m' + "Num. Ref. ", str(num_rows - 1), ": ", slope_l2, slope_h1, '\x1b[0m')
        assert round(slope_l2) >= round(float(method_order + 1.0))
        assert round(slope_h1) >= round(float(method_order))



def main():

    program_folder = str(Path(__file__).resolve().parent)

    dirpath_pcc_2d = program_folder + "/../Export"
    if os.path.exists(dirpath_pcc_2d) and os.path.isdir(dirpath_pcc_2d):
        shutil.rmtree(dirpath_pcc_2d)

    dirpath_pcc_2d = program_folder + "/../Export/Export_PCC_2D"
    if os.path.exists(dirpath_pcc_2d) and os.path.isdir(dirpath_pcc_2d):
        shutil.rmtree(dirpath_pcc_2d)

    tol = 1.0e-12
    test_type = 1
    method_orders = [1]

    method_types = [1, 1, 1, 2, 3]
    method_types_name = ["h_navem", "b_navem", "p_navem", "fem", "vem"]

    dictionary_files = [program_folder + '/../TrainedModels/H-NAVEM',
                        program_folder + '/../TrainedModels/B-NAVEM',
                        program_folder + '/../TrainedModels/P-NAVEM',
                        '', '']

    mesh_types = [5, 0]
    for mesh_type in mesh_types:
        mt = 0
        for method_type in method_types:
            print('\x1b[6;30;41m' + "Begin of convergence test with method_type =", method_types_name[mt] + '\x1b[0m')
            dictionary_file = dictionary_files[mt]
            for order in method_orders:
                export_path = dirpath_pcc_2d + "/Export_" + method_types_name[mt] + "_" + str(order) + "_" + str(
                    mesh_type)
                os.system(
                    "python " + program_folder + "/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1"
                    " --mesh-type={3} --mesh-max-relative-area=0.1 --export-path={2} --dictionary-file={4} "
                    "--no-evaluate-polynomial-loss --no-evaluate-laplacian-loss --no-evaluate-boundary-loss".format(
                        order, method_type, export_path, mesh_type, dictionary_file))
                os.system(
                    "python " + program_folder + "/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1 "
                    "--mesh-type={3} --mesh-max-relative-area=0.05 --export-path={2} --dictionary-file={4} "
                    "--no-evaluate-polynomial-loss --no-evaluate-laplacian-loss --no-evaluate-boundary-loss".format(
                        order, method_type, export_path, mesh_type, dictionary_file))
                os.system(
                    "python " + program_folder + "/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1 "
                    "--mesh-type={3} --mesh-max-relative-area=0.01 --export-path={2} --dictionary-file={4} "
                    "--no-evaluate-polynomial-loss --no-evaluate-laplacian-loss --no-evaluate-boundary-loss".format(
                        order, method_type, export_path, mesh_type, dictionary_file))
                os.system(
                    "python " + program_folder + "/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1 "
                    "--mesh-type={3} --mesh-max-relative-area=0.005 --export-path={2} --dictionary-file={4} "
                    "--no-evaluate-polynomial-loss --no-evaluate-laplacian-loss --no-evaluate-boundary-loss".format(
                        order, method_type, export_path, mesh_type, dictionary_file))

                errors = import_errors_pcc(export_path, method_type, order, test_type)
                test_errors_pcc(errors, order, tol)

            print("End of convergence test with method_type =", method_types_name[mt], "\n")
            mt += 1

    method_types = [1, 1, 1, 3]
    method_types_name = ["h_navem", "b_navem", "p_navem", "vem"]

    dictionary_files = [program_folder + '/../TrainedModels/H-NAVEM',
                        program_folder + '/../TrainedModels/B-NAVEM',
                        program_folder + '/../TrainedModels/P-NAVEM',
                        '']

    mesh_type = 4
    mesh_names = [program_folder + '/../Mesh/MVoro/MVoro_centroid_0.200000',
                  program_folder + '/../Mesh/MVoro/MVoro_centroid_0.100000',
                  program_folder + '/../Mesh/MVoro/MVoro_centroid_0.075000',
                  program_folder + '/../Mesh/MVoro/MVoro_centroid_0.040000']

    mt = 0
    for method_type in method_types:

        print('\x1b[6;30;41m' + "Begin of convergence test with method_type =", method_types_name[mt] + '\x1b[0m')
        dictionary_file = dictionary_files[mt]

        for order in method_orders:

            export_path = dirpath_pcc_2d + "/Export_" + method_types_name[mt] + "_" + str(order) + "_" + str(
                mesh_type)

            for mesh_name in mesh_names:

                os.system(
                    "python " + program_folder + "/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1"
                    " --mesh-type={3} --import-path={5} --export-path={2} --dictionary-file={4} "
                    "--no-evaluate-polynomial-loss --no-evaluate-laplacian-loss --no-evaluate-boundary-loss".format(
                        order, method_type, export_path, mesh_type, dictionary_file, mesh_name))


            errors = import_errors_pcc(export_path, method_type, order, test_type)
            test_errors_pcc(errors, order, tol)

        print("End of convergence test with method_type =", method_types_name[mt], "\n")
        mt += 1



if __name__ == '__main__':
    main()
