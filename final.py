#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script to test the accuracy of a netwerk with different values for the variables.

\nCreated on 1-3-2020

@author: N. Bijl
"""
from neuralNetwork import NeuralNetwork
from dataset import Dataset
import numpy as np 
import os 

# set constants/start values
# INPUTS  = 784
# OUTPUTS = 10
# HIDDEN  = 16
# LAYERS  = 3
# LR      = 0.02
ACT     = "sigmoid"

EPOCH   = 20
DATA    = 100
TEST    = 500

ds = Dataset()

def getAccuracy(var, begin, name, step, loop, path="./results/"):
    value = begin
    for i in range(loop):
        print(i)
        # create network
        if (var == 'hid'):
            nn = NeuralNetwork(hid=value, act=ACT)
        elif (var == 'lay'):
            nn = NeuralNetwork(lay=value, act=ACT)
        elif (var == 'lr'):
            nn = NeuralNetwork(lr=value, act=ACT)
        elif (var == 'epoch'):
            nn = NeuralNetwork(act=ACT)

        # train network
        if (var == 'epoch'):
            nn.train(ds.train_data, ds.train_labels_arr, value, DATA)
        else:
            nn.train(ds.train_data, ds.train_labels_arr, EPOCH, DATA)

        # counters
        correct = 0
        false   = 0
        for i in range(TEST):
            output = nn.feedforward(ds.test_data[i], isRound=False, isSoftmax=True)
            output_digit = np.where(output == np.amax(output))
            if output_digit[0][0] == ds.test_labels[i]:
                correct += 1
            else:
                false += 1
        # calc accuracy
        accuracy = (correct * 100) / (correct + false)
        # log accuracy
        logResults(path, name, value, accuracy)
        # increment value
        value += step

def logResults(path, name, value, accuracy):
    # set file path
    fpath = os.path.join(path, '{}.csv'.format(name))
    # check if dir exists
    if(not(os.path.exists(os.path.dirname(path)))):
        os.mkdir(os.path.dirname(path))

    with open(fpath, 'a') as fo:
        fo.write('{};{}\n'.format(
                value, 
                accuracy
        ))

# + == done

# getAccuracy('lr', 0.001, 'lr001', 0.001, 800)         # +
# getAccuracy('lr', 0.001, 'lr0001', 0.0001, 1000)      # +
# getAccuracy('hid', 1, 'hidden', 1, 300)               # +
# getAccuracy('lay', 1, 'layers', 1, 30)                # +
# getAccuracy('epoch', 100, 'epoch', 5, 1)              # +

ACT = 'tanh'

# getAccuracy('lr', 0.001, 'tlr001', 0.001, 800)        # +
# getAccuracy('lr', 0.001, 'tlr0001', 0.0001, 1000)     # +
# getAccuracy('hid', 1, 'thidden', 1, 300)              # +
# getAccuracy('lay', 1, 'tlayers', 1, 30)                # +/- after 25 error
# getAccuracy('epoch', 0, 'tepoch', 5, 21)              # +