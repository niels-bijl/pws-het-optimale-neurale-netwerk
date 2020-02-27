#!/usr/bin/env python3

from neuralNetwork10 import NeuralNetwork
from dataset2 import Dataset
import numpy as np
import os
import csv


def logResults(path, accuracy):
    # check if dir exists
    if(not(os.path.exists(os.path.dirname(path)))):
        os.mkdir(os.path.dirname(path))

    with open(path, 'a', newline='') as file:
        fo = csv.writer(file, delimiter=';')
        fo.writerow(["Hidden","Layers","Lr","Act","Epoch","Data","Test","Accuracy"])
        fo.writerow([HIDDEN, LAYERS, LR, ACT, EPOCH, DATA, TEST, accuracy])
        fo.writerow([])

# dir to save file to
PATH = "./results/results.csv"
PATH_A = "./results/highest_accuracy.text"

# set constants
INPUTS  = 784
OUTPUTS = 10
HIDDEN  = 16
LAYERS  = 3
LR      = 0.02
ACT     = "sigmoid"
# ACT     = "tanh"

EPOCH   = 20
DATA    = 60
TEST    = 10000

# create counter
correct = 0
false   = 0 

# init dataset
ds = Dataset()
# init neural network
nn = NeuralNetwork(INPUTS, OUTPUTS, HIDDEN, LAYERS, LR, ACT)

header = "HIDDEN: {}; LAYERS:{}; LR:{}; ACT:{}; EPOCH:{}; DATA:{}; TEST:{}".format(HIDDEN, LAYERS, LR, ACT, EPOCH, DATA, TEST)

print("-" * 100)
print(header)
print("-" * 100)

nn.train(ds.train_data, ds.train_labels_arr, EPOCH, DATA)
nn.logwb()
#nn.setwb()

for i in range(TEST):
    output_data = nn.feedforward(ds.test_data[i], False)
    output_digit = np.where(output_data == np.amax(output_data))
    if output_digit[0][0] == ds.test_labels[i]:
        correct += 1
    else:
        false += 1
    #print(nn.feedforward(ds.train_data[i], False), output_digit[0], ds.train_labels[i])

# calc accuracy
accuracy = (correct * 100) / (correct + false)

print("-" * 100)
print("accuracy: {}%".format(accuracy))
print("-" * 100)

#logResults(PATH, accuracy)