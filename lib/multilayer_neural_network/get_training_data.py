#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
def liner_training_data(x_range, y_range, split=10):
    train_data = {}
    xspan = float(x_range[1] - x_range[0]) / split
    yspan = float(y_range[1] - y_range[0]) / split
    for x_value in [ float(i)*xspan+x_range[0] for i in range(split)]:
        collect_value = x_value
        for y_value in [ float(j) * yspan + y_range[0]  for j in range(split)]:
            classification = 1 if y_value >= collect_value else 0
            train_data[(x_value, y_value)] = classification
    return train_data

def quadratic_function_data(x_range, y_range, split=10):
    train_data = {}
    xspan = float(x_range[1] - x_range[0]) / split
    yspan = float(y_range[1] - y_range[0]) / split
    for x_value in [ float(i)*xspan+x_range[0] for i in range(split)]:
        collect_value = (x_value ) ** 3
        for y_value in [ float(j) * yspan + y_range[0]  for j in range(split)]:
            classification = 1 if y_value >= collect_value else 0
            train_data[(x_value, y_value)] = classification

    return train_data


def sin_function_data(x_range, y_range, split=10):
    train_data = {}
    xspan = float(x_range[1] - x_range[0]) / split
    yspan = float(y_range[1] - y_range[0]) / split
    height = float(y_range[1] - y_range[0]) /3

    for x_value in [ float(i)*xspan+x_range[0] for i in range(split)]:
        collect_value = math.sin(x_value*np.pi*2) * height + float(y_range[1] - y_range[0]) /2
        for y_value in [ float(j) * yspan + y_range[0]  for j in range(split)]:
            classification = 1 if y_value >= collect_value else 0
            train_data[(x_value, y_value)] = classification

    return train_data


def change_format(data):
    train_data_input  = []
    train_data_output = []
    for xy_value, classification in data.items():
        train_data_input.append(np.array([ xy_value[0],xy_value[1] ] ) )
        train_data_output.append(np.array([ classification ]) )
    return np.array(train_data_input) , np.array(train_data_output)
