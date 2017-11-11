#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import identifier
import evaluator


def main():
    model = identifier.train_network()

    print("Result: {}".format(evaluator.evaluate_id(model)))


if __name__ == "__main__":
    main()
