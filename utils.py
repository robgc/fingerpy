# -*- coding: utf-8 -*-

import glob

import numpy as np


def get_input_training_data() -> np.ndarray:
    """
    Retrieve training data from fingerprint database.

    :return: A training data set as a (n_training_files x 256) matrix.
    """
    data_rows = list()

    # The for loop is necessary if we want to read the fingerprints in order

    for i in range(1, 112 + 1):
        path_str = "rawData/QFM16_" + str(i) + "_[3-5]_*.txt"
        fp_paths = glob.glob(path_str)

        for fp in fp_paths:
            fp_data = read_finger_file(fp)
            data_rows.append(fp_data)

    result = np.array(data_rows)

    return result


def get_id_output_training_data() -> np.ndarray:
    """
    Returns a vector with the category of each fingerprint file. In this case we
    have 112 categories (1 for each finger). Categories go from 0 to 111,
    [0, 112).

    :return: A vector ready to be used with <keras.utils.to_categorical>
    """
    data = list()

    # The for loop is necessary if we want to read the fingerprints in order

    for i in range(1, 112 + 1):
        path_str = "rawData/QFM16_" + str(i) + "_[3-5]_*.txt"
        fp_paths = glob.glob(path_str)

        data += [i-1] * len(fp_paths)

    result = np.array(data)

    return result


def get_input_testing_data() -> np.ndarray:
    """
    Retrieve testing data from the database.

    :return: A testing data set as a (n_testing_files x 256) matrix.
    """
    data = list()

    for i in range(1, 112 + 1):
        for j in range(1, 3):
            path_str = "rawData/QFM16_" + str(i) + "_" + str(j) + "_*.txt"
            fp_paths = glob.glob(path_str)

            for path in fp_paths:
                fp_data = read_finger_file(path)
                data.append(fp_data)

    result = np.array(data)

    return result


def get_id_output_testing_data() -> np.ndarray:
    """
    Retrieve the expected categorical output from the testing data. We
    have 112 categories (1 for each finger). Categories go from 0 to 111,
    [0, 112).

    :return: A vector ready to be used with <keras.utils.to_categorical>
    """
    data = list()

    # The for loop is necessary if we want to read the fingerprints in order

    for i in range(1, 112 + 1):
        path_str = "rawData/QFM16_" + str(i) + "_[1-2]_*.txt"
        fp_paths = glob.glob(path_str)
        data += [i-1] * len(fp_paths)

    result = np.array(data)

    return result


# def select_fp(fp_paths: list) -> np.ndarray:
#     sum_values = list()
#     diffs = list()
#
#     for fp in fp_paths:
#         fp_data = read_finger_file(fp)
#         fp_sum = np.sum(fp_data)
#         sum_values.append(fp_sum)
#
#     average = np.average(sum_values)
#
#     for val in sum_values:
#         diffs.append(abs(val - average))
#
#     min_index = np.argmin(diffs)
#
#     return fp_paths[min_index]


def read_fingerprint(finger_name: str) -> np.ndarray:
    """
    Given the file "x_y_z" name this function returns a vector with
    the fingerprint data.

    :param finger_name: A string with the format "x_y_z".
    :return: A vector (1x256) containing the fingerprint data.
    """

    base_path = "rawData/QFM16_"
    path = base_path + finger_name + ".txt"

    return read_finger_file(path)


def read_finger_file(path: str) -> np.ndarray:
    """
    Given the path of the .txt file this function returns
    the vector with the fingerprint data.

    :param path: The fingerprint file path.
    :return: A vector (1x256) with the fingerprint info.
    """

    finger_file = open(path, "r")
    data_rows = list()

    for line in finger_file.readlines():
        data_rows.append([int(x) for x in line.split(" ")[:-1]])

    finger_file.close()

    result = np.array(data_rows).flatten()

    return result
