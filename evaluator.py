#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import numpy as np
import glob

import utils


def evaluate_id(model):
    # Get testing data

    x_test = utils.get_input_testing_data()
    y_test = keras.utils.to_categorical(utils.get_id_output_testing_data(),
                                        num_classes=112)

    # Making predictions using each window from the samples

    w_results = list()

    for i in range(len(x_test)):
        x_reshaped = np.reshape(x_test[i], (1, 256))
        prediction = model.predict(x_reshaped)

        r_predicted = np.argmax(prediction) + 1
        r_prob = np.max(prediction)

        w_results.append([r_predicted, r_prob])

    w_results = np.array(w_results)

    # Count the number of samples and correct samples

    index = 0
    n_samples = 0
    n_correct = 0

    for i in range(1, 113):  # Number of fingers
        for j in range(1, 3):  # Number of samples
            # Retrieve number of windows for this sample

            n_ws = len(glob.glob("rawData/QFM16_" + str(i) + "_" + str(j) +
                                 "_*.txt"))

            # For this sample keep the prediction with the highest probability

            w_max_prob = -1
            w_max_val = -1

            for k in range(index, index + n_ws):
                cur_val = w_results[k][0]
                cur_prob = w_results[k][1]

                if cur_prob > w_max_prob:
                    w_max_prob = cur_prob
                    w_max_val = cur_val

            # If the prediction with the highest probability matches with the
            # expected finger then we have a correct answer

            if i == w_max_val:
                n_correct += 1

            n_samples += 1
            index += n_ws  # Place the index on the next window set

    result = n_correct/n_samples

    return result
