# -*- coding: utf-8 -*-

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras import metrics

import utils


def train_network() -> Sequential:
    x_training_data = utils.get_input_training_data()
    y_training_data = keras.utils.to_categorical(
        utils.get_id_output_training_data(), num_classes=112)

    model = Sequential()
    model.add(Dense(112, activation="relu", input_dim=256))
    model.add(Dense(112, activation="relu"))
    model.add(Dense(112, activation="softmax"))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.85)

    model.compile(loss="categorical_crossentropy", optimizer=sgd,
                  metrics=[metrics.categorical_accuracy])

    model.fit(x_training_data, y_training_data, epochs=500, batch_size=1071)

    return model
