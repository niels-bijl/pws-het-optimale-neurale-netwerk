#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script to get the accuracy of a network.

\nAdded softmax

\nCreated on 1-3-2020

@author: N. Bijl
"""

from neuralNetwork import NeuralNetwork
from dataset import Dataset
import numpy as np 
import os 

# set constants/start values
INPUTS  = 784
OUTPUTS = 10
HIDDEN  = 131
LAYERS  = 4
LR      = 0.0595
ACT     = "sigmoid"

EPOCH   = 65
DATA    = 100
TEST    = 500

ds = Dataset()

nn = NeuralNetwork(INPUTS, OUTPUTS, HIDDEN, LAYERS, LR, ACT)

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
# set file path
fpath = os.path.join('./results/', 'opt.csv')

with open(fpath, 'a') as fo:
    fo.write('{};{};{};{};{};{};{};{};{};{}\n'.format(
            INPUTS,
            OUTPUTS, 
            HIDDEN,
            LAYERS,
            LR,
            ACT,
            EPOCH,
            DATA,
            TEST,
            accuracy
    ))
    