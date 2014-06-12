#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer

from lib.data_source.get_training_data import liner_training_data,quadratic_function_data, sin_function_data, change_format

def build_network():
    n = FeedForwardNetwork()

    inLayer = LinearLayer(2)
    hiddenLayer = SigmoidLayer(5)
    outLayer = LinearLayer(1)
    #outLayer = SigmoidLayer(1)

    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)

    in2hidden = FullConnection(inLayer, hiddenLayer)
    hidden2out = FullConnection(hiddenLayer, outLayer)
    n.addConnection(in2hidden)
    n.addConnection(hidden2out)

    n.sortModules()
    return n

def get_supervised(network, train_data_input, train_data_output):
    supervised = SupervisedDataSet(network.indim, 1)
    for i in range(len(train_data_input)):
        supervised.addSample(train_data_input[i], train_data_output[i])
    return supervised

def test_multilayer_perceptron():

    def plot(fig, data):
        ax  = fig.add_subplot(111)
        ax.plot([x[0] for x in data], [x[1] for x in data])

    def scat(fig, liner_data, marker='o', color='g'):
        ax  = fig.add_subplot(111)
        ax.scatter([x[0] for x in liner_data], [x[1] for x in liner_data], marker=marker, color=color, s=10)

    def get_predict_list(x_range, y_range, nn, split=10):
        data = []
        xspan = float(x_range[1] - x_range[0]) / split
        yspan = float(y_range[1] - y_range[0]) / split

        for x_value in [ float(i)*xspan+x_range[0] for i in range(split)]:
            predict_list = []
            for y_value in [ float(j) * yspan + y_range[0]  for j in range(split)]:
                #if nn.predict([x_value,y_value])[0] >= 0.5:
                if nn.activate([x_value,y_value])[0] >= 0.5:
                    data.append((x_value, y_value))
                    break
        return data

    import matplotlib.pyplot as plt

    """ トレーニングデータ取得
    """
    x_range = [0,1]
    y_range = [0,1]
    #liner_data = liner_training_data(x_range, y_range)
    liner_data = quadratic_function_data(x_range, y_range, split=20)
    #liner_data = sin_function_data(x_range, y_range, 20)
    train_data_input, train_data_output = change_format(liner_data)

    fig = plt.figure()
    scat(fig, [key for key, value in liner_data.items() if value == 0], color='g' )
    scat(fig, [key for key, value in liner_data.items() if value == 1], color='b' )



    """ NN構築
    """
    network = build_network()

    # mlnn = MultiLayerNeuralNetwork( [2, 5, 1],
    #                                 threshold=0.1,
    #                                 start_learning_coef=0.2,
    #                                 sigmoid_alpha=10,
    #                                 mini_batch=100,
    #                                 layer_type=[LinearLayer, SigmoidLayer, SigmoidLayer],
    #                                 rprop=True
    #                                 )

    """ 学習
    """
    #error_hist = mlnn.train_multi(train_data_input, train_data_output)
    supervised = get_supervised(network, train_data_input, train_data_output)
    trainer = RPropMinusTrainer(network, dataset=supervised, batchlearning=True, verbose=True)
    trainer.trainUntilConvergence(maxEpochs=100)


    # xに対応するyを算出, 学習後分離線書く
    data = get_predict_list(x_range,y_range, network, split=20)
    plot(fig, data)

    # # エラー表示
    # fig2 = plt.figure()
    # plot(fig2, error_hist)

    # 表示
    plt.show()


if __name__ == '__main__':
    test_multilayer_perceptron()
