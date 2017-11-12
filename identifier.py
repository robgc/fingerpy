# -*- coding: utf-8 -*-

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras import metrics
import glob
import numpy as np

import utils


def train_network() -> Sequential:
    x_training_data = utils.get_input_training_data()
    y_training_data = keras.utils.to_categorical(
        utils.get_id_output_training_data(), num_classes=112)

    model = Sequential()
    model.add(Dense(256, activation="sigmoid", input_dim=256))
    model.add(Dense(256, activation="linear"))
    model.add(Dense(112, activation="softmax"))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.95)

    model.compile(loss="categorical_crossentropy", optimizer=sgd,
                  metrics=[metrics.categorical_accuracy])

    model.fit(x_training_data, y_training_data, epochs=600, batch_size=1480)

    return model


def evaluate(model):
    # Get testing data

    x_test = utils.get_input_testing_data()

    # Making predictions using each window from the samples

    w_results = list()

    for x in x_test:
        x_reshaped = np.reshape(x, (1, 256))
        prediction = model.predict(x_reshaped)

        r_predicted = np.argmax(prediction) + 1
        r_prob = np.max(prediction)

        w_results.append([r_predicted, r_prob])

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
