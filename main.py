#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import identifier
import authenticator
import utils
import numpy as np


def main():
    print("Training the fingerprint identificator network")
    model = identifier.train_network()
    print("Result: {}".format(identifier.evaluate(model)))

    # print("=============")
    #
    # print("Training the fingerprint authenticator network")
    # x_train = utils.get_input_testing_data()
    # y_train = utils.get_auth_output_testing_data()
    # model = authenticator.train_network()
    # print("Result: {}".format(model.evaluate(x_train, y_train,
    #                                          batch_size=1000)))
    # x_ts = np.random.randint(17, size=(1, 256))
    # print(x_ts)
    # print(model.predict(x_ts))


if __name__ == "__main__":
    main()
