import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import shutil


def import_errors_mcc(export_path, method_type, method_order, test_type):
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
                errors_row.append(row[7])  # h
                errors_row.append(row[8])  # error l2 pressure
                errors_row.append(row[9])  # error l2 velocity
                errors_row.append(row[10])  # error l2 super pressure
                errors_row.append(row[11])  # norm l2 pressure
                errors_row.append(row[12])  # norm l2 velocity
            else:
                errors_row.append(float(row[7]))
                errors_row.append(float(row[8]))
                errors_row.append(float(row[9]))
                errors_row.append(float(row[10]))
                errors_row.append(float(row[11]))
                errors_row.append(float(row[12]))
            errors.append(errors_row)
            counter += 1

    return errors


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


def test_errors_mcc(errors, method_order, tolerance):
    num_rows = len(errors)

    if num_rows == 2:
        print('\x1b[0;31;40m' + "Num. Ref. 1: ", abs(errors[1][1]) / abs(errors[1][3]),
              abs(errors[1][2]) / abs(errors[1][4]), '\x1b[0m')
        assert abs(errors[1][1]) < tolerance * abs(errors[1][3])
        assert abs(errors[1][2]) < tolerance * abs(errors[1][4])
    else:
        errors = np.array(errors[1:])
        slope_l2_press = float(np.polyfit(np.log(errors[:, 0]), np.log(errors[:, 1]), 1)[0])
        slope_l2_vel = float(np.polyfit(np.log(errors[:, 0]), np.log(errors[:, 2]), 1)[0])
        slope_l2_press_super = float(np.polyfit(np.log(errors[:, 0]), np.log(errors[:, 3]), 1)[0])
        print('\x1b[0;31;40m' + "Num. Ref. ", str(num_rows - 1), ": ", slope_l2_press, slope_l2_vel,
              slope_l2_press_super, '\x1b[0m')
        assert round(slope_l2_press) >= round(float(method_order))
        assert round(slope_l2_press_super) >= round(float(method_order + 1))
        assert round(slope_l2_vel) >= round(float(method_order))


dirpath_pcc_2d = "./Export/Export_PCC_2D"
if os.path.exists(dirpath_pcc_2d) and os.path.isdir(dirpath_pcc_2d):
    shutil.rmtree(dirpath_pcc_2d)


def main():
    tol = 1.0e-12
    test_type = 1
    method_orders = [1]
    method_types = [1, 1, 1, 2, 3]
    method_types_name = ["h_navem", "b_navem", "p_navem", "fem", "vem"]
    dictionary_files = ['./TrainedModels/H-NAVEM/dictionary.txt', './TrainedModels/B-NAVEM/dictionary.txt',
                        './TrainedModels/P-NAVEM/dictionary.txt', '', '']
    mesh_types = [5, 0]
    for mesh_type in mesh_types:
        mt = 0
        for method_type in method_types:
            print("Begin of convergence test with method_type =", method_types_name[mt])
            dictionary_file = dictionary_files[mt]
            for order in method_orders:
                export_path = dirpath_pcc_2d + "/Export_" + method_types_name[mt] + "_" + str(order) + "_" + str(
                    mesh_type)
                os.system(
                    "python ./examples/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1 --mesh-type={3} --mesh-max-relative-area=0.1 --export-path={2} --dictionary-file={4}".format(
                        order, method_type, export_path, mesh_type, dictionary_file))
                os.system(
                    "python ./examples/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1 --mesh-type={3} --mesh-max-relative-area=0.05 --export-path={2} --dictionary-file={4}".format(
                        order, method_type, export_path, mesh_type, dictionary_file))
                os.system(
                    "python ./examples/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1 --mesh-type={3} --mesh-max-relative-area=0.01 --export-path={2} --dictionary-file={4}".format(
                        order, method_type, export_path, mesh_type, dictionary_file))
                os.system(
                    "python ./examples/main_elliptic_pcc_2d.py --method-order={0} --method-type={1} --test-id=1 --mesh-type={3} --mesh-max-relative-area=0.005 --export-path={2} --dictionary-file={4}".format(
                        order, method_type, export_path, mesh_type, dictionary_file))

                errors = import_errors_pcc(export_path, method_type, order, test_type)
                test_errors_pcc(errors, order, tol)

            print("End of convergence test with method_type =", method_types_name[mt], "\n")
            mt += 1


if __name__ == '__main__':
    main()
