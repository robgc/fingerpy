#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import identifier
import authenticator


def main():
    print("Testing the fingerprint identifier network")
    results = list()
    for i in range(10):
        model = identifier.train_network()
        result = identifier.evaluate(model)
        results.append(result)
    print("Result: {}".format(sum(results)/len(results)))

    print("=============")

    print("Testing the fingerprint authenticator network")
    results = list()
    for i in range(10):
        model = authenticator.train_network()
        result = authenticator.evaluate(model)
        results.append(result)
    print("Result: {}".format(sum(results)/len(results)))


if __name__ == "__main__":
    main()
