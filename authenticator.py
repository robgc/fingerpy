#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import glob

import utils


def train_network():
    x_training_data = utils.get_input_training_data()
    y_training_data = utils.get_auth_output_training_data()

    model = Sequential()
    model.add(Dense(256, activation="sigmoid", input_dim=256))
    model.add(Dense(256, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)

    model.compile(loss="binary_crossentropy", optimizer=sgd,
                  metrics=["accuracy"])

    model.fit(x_training_data, y_training_data, epochs=400, batch_size=1480)

    return model


def evaluate(model):
    # Get testing data

    x_test = utils.get_input_testing_data()

    # Making predictions using each window from the samples

    w_results = list()

    for x in x_test:
        x_reshaped = np.reshape(x, (1, 256))
        prediction = model.predict(x_reshaped)

        r_prob = np.max(prediction)

        w_results.append(r_prob)

    w_results = np.array(w_results)

    # Count the number of samples and correct predictions

    index = 0
    n_samples = 0
    n_correct = 0

    for i in range(1, 113):  # Number of fingers
        # Retrieve number of windows for this sample

        n_ws = len(glob.glob("rawData/QFM16_" + str(i) + "_5_*.txt"))

        if n_ws != 0:
            # For this sample keep the prediction with the highest probability

            w_max_prob = -1

            for k in range(index, index + n_ws):
                cur_prob = w_results[k]

                if cur_prob > w_max_prob:
                    w_max_prob = cur_prob

            # If the prediction with the highest probability matches with the
            # expected result then we have a correct answer

            if (i < 6 and w_max_prob >= 0.9) or (i >= 6 and w_max_prob < 0.9):
                n_correct += 1

            n_samples += 1
            index += n_ws  # Place the index on the next window set

    result = n_correct / n_samples

    return result
