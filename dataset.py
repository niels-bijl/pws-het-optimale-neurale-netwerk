#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script to load the MNIST dataset.

\nCreated on 1-3-2020

@author: N. Bijl
"""
import gzip
import numpy as np
import os

class Dataset:
    def __init__(self):
        self.train_data      =   self.loadImages('train-images-idx3-ubyte.gz')
        self.train_labels    =   self.loadLabels('train-labels-idx1-ubyte.gz')
        self.test_data       =   self.loadImages('t10k-images-idx3-ubyte.gz')
        self.test_labels     =   self.loadLabels('t10k-labels-idx1-ubyte.gz')

        self.train_labels_arr   = self.labelToArr(self.train_labels)
        self.test_labels_arr    = self.labelToArr(self.test_labels)

    def download(self, filename, source="http://yann.lecun.com/exdb/mnist/"):
        print("Downloading ", filename)
        import urllib.request as url
        url.urlretrieve(source+filename,("./mnist/" + filename))
        url.urlcleanup()

    def checkDir(self):
        if not os.path.exists("./mnist"):
            os.mkdir("./mnist")

    def loadImages(self, filename):
        self.checkDir()
        if (not(os.path.exists("./mnist/" + filename))):
            self.download(filename)

        with gzip.open("./mnist/" + filename,'rb') as fo:
            data = np.frombuffer(fo.read(), np.uint8, offset=16)
            data = data.reshape(-1, 784)

            return data/np.float64(256)

    def loadLabels(self, filename):
        self.checkDir()
        if (not(os.path.exists("./mnist/" + filename))):
            self.download(filename)

        with gzip.open("./mnist/" + filename, 'rb') as fo:
            data = np.frombuffer(fo.read(), np.uint8, offset=8)
            return data

    def labelToArr(self, labels, length=10):
        labels_re = np.zeros((len(labels), length))
        for i in range(len(labels)):
            labels_re[i][labels[i]] = 1
        return labels_re    
