# -*- coding: utf-8 -*-

import numpy as np


def read_fingerprint(finger_name: str) -> np.ndarray:
    """
    Given the finger name this function returns a matrix with
    the fingerprint data.
    :param finger_name: A string with the format "i_j_k".
    :return: A matrix containing the fingerprint data.
    """

    base_path = "rawData/QFM16_"
    path = base_path + finger_name + ".txt"

    return read_finger_file(path)


def read_finger_file(path: str) -> np.ndarray:
    """
    Given the path of the .txt file this function returns
    the matrix with the fingerprint data.

    :param path: The fingerprint info file path.
    :return: A matrix with the fingerprint info.
    """

    finger_file = open(path, "r")
    data_rows = list()

    for line in finger_file.readlines():
        data_rows.append([int(x) for x in line.split(" ")[:-1]])

    result = np.array(data_rows)

    return result
