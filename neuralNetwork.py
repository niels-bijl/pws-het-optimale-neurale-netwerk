#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neuralNetwork.py is a module to create a neural network with variable 
inputs, outputs, hidden neurons, hidden layers, learning rate and activation function.
The activation function is either sigmoid or tanh, any similar function could be added.
\nThis module is not optimized. 

\nAdded softmax

\nCreated on Mon Dec  10 13:30:30 2019

@author: N. Bijl, O. Erkemeij
"""

import numpy as np
import os

class NeuralNetwork:
    """
    Class to generate a neural network.\n
    inp: input neurons\n 
    out: output neurons\n 
    hid: hidden neurons(default=1)\n
    lay: hidden_layers(default=1)\n
    lr:  learning_rate(default=0.1)\n 
    act: activation_function(default="sigmoid")
    """
    def __init__(self, inp=784, out=10, hid=16, lay=3, lr=0.02, act="sigmoid", pathwb="./logwb"):
        """Method to create an instance of the neuralNetwork class."""
        self.inputs = inp
        self.outputs = out
        self.hidden = hid
        self.layers = lay
        self.lr = lr
        self.act = act

        # set path to dir for log file
        self.pathwb = os.path.join(pathwb, "wbi{}o{}h{}l{}l{}a{}".format(self.inputs, self.outputs, self.hidden, self.layers, self.lr, self.act))
        
        # seed to generate the same random values every time.
        np.random.seed(0)
        # generate arrays with random weights
        self.weights_ih = np.random.randn(self.hidden, self.inputs)

        if(self.layers > 1):
            self.weights_hh = np.random.randn((self.layers -1), self.hidden, self.hidden)
        
        self.weights_ho = np.random.randn(self.outputs, self.hidden)

        # generate arrays with random biases
        self.bias_h = np.random.randn(self.layers, self.hidden)
          
        self.bias_o = np.random.randn(self.outputs)
        
        # set the desired activation function
        if(self.act == "sigmoid" or self.act == "tanh"):
            if(self.act == "sigmoid"):
                # activation function of sigmoid
                act_func = lambda x: 1 / (1 + np.exp(-x))
                # derivative of sigmoid
                act_d_func = lambda y: y * (1 - y)
            elif(self.act == "tanh"):
                # activation function of tanh
                act_func = lambda x: np.tanh(x)
                # derivative of tanh
                act_d_func = lambda y: 1 - (y * y)

            # vectors are made to use a function on a array 
            # make vector of activation function
            self.act = np.vectorize(act_func)
            # make vector of derivative of the activation function
            self.act_d = np.vectorize(act_d_func)
        else:
            print("Warning, {} is not a known activation function".format(act))
            return
        
    def feedforward(self, inputs, isRound=False, isSoftmax=True):
        """
        Method to feedforward an array of inputs to calculate the outputs.\n
        inputs: a numpy array with the inputs.\n
        isRound: a boolean function to convert the outputs to rounded values(default=False).
        """
        # create an array to store the calculated hidden outputs
        hidden = []
        # calculate the output of the first hidden layer
        # output: act(dot(weights, inputs) + bias)
        # dot: the dot product of two matrices
        hidden.append(np.dot(self.weights_ih, inputs))
        hidden[0] += self.bias_h[0]
        hidden[0] = self.act(hidden[0])      
        
        # calculate the output of the hidden layers if there is more than 1 hidden layer
        # output: act(dot(weights, output of previous hidden layer) + bias)
        if(self.layers > 1):
            for i in range(self.layers -1):
                i += 1
                hidden.append(np.dot(self.weights_hh[i -1], hidden[i -1]))
                hidden[i] += self.bias_h[i]
                hidden[i] = self.act(hidden[i])  

        # calculate the output of the output layer
        # output: act(dot(weights, output of last hidden layer) + bias)    
        outputs = np.dot(self.weights_ho, hidden[-1])
        outputs += self.bias_o
        outputs = self.act(outputs)
        
        # round the outputs if desired
        if(isSoftmax == True):
            outputs = self.softmax(outputs)
        if(isRound == True):
            round_func = lambda r: round(r)
            round_vect = np.vectorize(round_func)
            outputs = round_vect(outputs)
        
        return outputs

    def softmax(self, outputs):
        """Method to return the normalized output."""
        # output[i] / sum
        sum = np.sum(outputs)
        outputs = outputs/sum

        return outputs

    
    def train(self, inputs_arr, targets_arr, epochs=1, data=None):
        """
        Method to train the neural network by adjusting the weights and biases.\n
        inputs: a numpy array with the inputs.\n 
        targets: a numpy array with the targets that belong to the inputs.
        """

        # assertion
        assert data <= len(inputs_arr)
        # bach is equel to the array length if it is not set
        if(data == None):
            data = len(inputs_arr)

        # iterate for batch for e epochs
        for e in range(epochs):
            for d in range(data): 
                inputs = inputs_arr[d]
                targets = targets_arr[d]
                # calculate the output (see feedforward)
                # feedforward is not used in this function because the outputs of the hidden layers are needed to calculate the new weights
                hidden = []
                hidden.append(np.dot(self.weights_ih, inputs))
                hidden[0] += self.bias_h[0]
                hidden[0] = self.act(hidden[0])      
                
                if(self.layers > 1):
                    for i in range(self.layers -1):
                        i += 1
                        hidden.append(np.dot(self.weights_hh[i -1], hidden[i -1]))
                        hidden[i] += self.bias_h[i]
                        hidden[i] = self.act(hidden[i])  
                    
                outputs = np.dot(self.weights_ho, hidden[-1])
                outputs += self.bias_o
                outputs = self.act(outputs)
                
                # backpropagation
                # calculate error (output --> input)
                output_error = targets - outputs
                

                # transpose weights (output --> input)
                # the weights are transposed because of the network going backwards
                weights_hot = np.transpose(self.weights_ho)

                weights_hht = []
                for i in range(self.layers -1):
                    weights_hht.append(np.transpose(self.weights_hh[-i-1]))
                    
                # calculate the error of the hidden layers (output --> input)
                hidden_error = []
                hidden_error.append(np.dot(weights_hot, output_error))
                for i in range(self.layers -1):
                    hidden_error.append(np.dot(weights_hht[i], hidden_error[i]))
                
                # weights =  weights + lr * error * derivative of act * outputs of the layers transposed
                # calculate activation derivative (output --> input)
                output_s = self.act_d(outputs)
                
                hidden_s = []
                for i in range(self.layers):
                    hidden_s.append(self.act_d(hidden[-i-1]))      
                
                # get transposed outputs of layers (output --> inputs)
                hidden_t = []
                for i in range(self.layers):
                    hidden_t.append(np.transpose(hidden[-i-1])[np.newaxis])
                    
                inputs_t = np.transpose(inputs)[np.newaxis]
            
                # calculate the gradient (output-->input)
                # gradient = error * derivative
                outputs_g = output_error * output_s
                
                hidden_g = []
                for i in range(self.layers):
                    hidden_g.append(hidden_error[i] * hidden_s[i])   
                
                # reshape gradient (output-->input)
                # reshape is needed for numpy to calculate the dot product
                outputs_gr = np.reshape(outputs_g,(self.outputs,1))
                
                hidden_gr = []
                for i in range(self.layers):
                    hidden_gr.append(np.reshape(hidden_g[i],(self.hidden,1)))
                
                # add delta weights to weights (output-->hidden)
                # delta weights = dot(gradient, transposed outputs of previous layer)
                # dot: dot product of two matrices
                delta_weights_ho = np.dot(outputs_gr[0], hidden_t[0])
                delta_weights_ho *= self.lr
                self.weights_ho += delta_weights_ho
                
                delta_weights_hh = []
                if (self.layers > 1):
                    for i in range(self.layers -1):
                        delta_weights_hh.append((np.dot(hidden_gr[i], hidden_t[-i-1]) * self.lr))
                        self.weights_hh[-i-1] += delta_weights_hh[i]
                
                delta_weights_ih = np.dot(hidden_gr[-1], inputs_t)
                delta_weights_ih *= self.lr
                self.weights_ih += delta_weights_ih
                
                # add gradient to bias (ouput-->input)
                self.bias_o += (outputs_g * self.lr)
                for i in range(self.layers):
                    self.bias_h[-i-1] += (hidden_g[i] * self.lr)

                # print num of done epochs
                # print num of done data
                print("data: {:5}/{}\t\t epochs: {:5}/{}".format(d+1, data, e+1, epochs), end="\r") 
        # print new line
        print("")
    
    def logwb(self):
        """Method to log the weights and biases."""
        # check if the dir exists
        if(not(os.path.exists(os.path.dirname(self.pathwb)))):
            print(os.path.dirname(self.pathwb))
            os.mkdir(os.path.dirname(self.pathwb))

        # set file name and its path
        filename_ih = "wih.txt"
        path_wih = os.path.join(self.pathwb, filename_ih)

        # make array with weights and biases for each layer
        if (self.layers > 1):
            path_whh = []
            for i in range(self.layers -1):
                path_whh.append(os.path.join(self.pathwb, "whh{}.txt".format(i)))

        filename_ho = "who.txt"
        path_who = os.path.join(self.pathwb, filename_ho)

        path_bh = []
        for i in range(self.layers):
            path_bh.append(os.path.join(self.pathwb, "bh{}.txt".format(i)))
        
        filename_bo = "bo.txt"
        path_bo = os.path.join(self.pathwb, filename_bo)

        if(not(os.path.exists(self.pathwb))):
            os.mkdir(self.pathwb)
            # create file for ih
            with open(path_wih, 'w') as fo:
                # add data tab spaced, end of line \n
                for d in range(self.hidden):
                    for i in range(self.inputs):
                        fo.write("{}\t".format(self.weights_ih[d][i]))
                    # create newline
                    fo.write("\n")
            # create file for hh
            if(self.layers > 1):
                for k in range(self.layers - 1):
                    with open(path_whh[k], 'w') as fo:
                        # add data tab spaced, end of line \n
                        for d in range(self.hidden):
                            for i in range(self.hidden):
                                fo.write("{}\t".format(self.weights_hh[k][d][i]))
                        # create newline
                        fo.write("\n")
            # create file for ho
            with open(path_who, 'w') as fo:
                # add data tab spaced, end of line \n
                for d in range(self.outputs):
                    for i in range(self.hidden):
                        fo.write("{}\t".format(self.weights_ho[d][i]))
                    # create newline
                    fo.write("\n")
            # same for biases
            for i in range(self.layers):
                with open(path_bh[i], 'w') as fo:
                    # add data tab spaced, end of line \n
                    for j in range(self.hidden):
                        fo.write("{}\t".format(self.bias_h[i][j]))
                    # create newline
                    fo.write("\n")
            with open(path_bo, 'w') as fo:
                # add data tab spaced, end of line \n
                for i in range(self.outputs):
                    fo.write("{}\t".format(self.bias_o[i]))
                # create newline
                fo.write("\n")

    def loadwb(self):
        """Method to load loged weights and biases"""
        wih = np.loadtxt(os.path.join(self.pathwb, "wih.txt"))

        # create array to store weights and biases and load the weights and biases
        if(self.layers > 1):
            # create np array
            whh = np.zeros((self.layers -1, self.hidden, self.hidden))
            for i in range(self.layers -1):
                whh[i] = np.loadtxt(os.path.join(self.pathwb, "whh{}.txt".format(i))).reshape(16,16)

        who = np.loadtxt(os.path.join(self.pathwb, "who.txt"))

        bh = np.zeros((self.layers, self.hidden))
        for i in range(self.layers):
            bh[i] = np.loadtxt(os.path.join(self.pathwb, "bh{}.txt".format(i)))

        bo = np.loadtxt(os.path.join(self.pathwb, "bo.txt"))

        return wih, whh, who, bh, bo

    def setwb(self):
        # set the loaded weights and biases
        self.weights_ih, self.weights_hh, self.weights_ho, self.bias_h, self.bias_o = self.loadwb()
        
