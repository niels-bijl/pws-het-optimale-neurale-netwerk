#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script to plot result from a .csv file.

\nCreated on 1-3-2020

@author: N. Bijl
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np 

VAR = 'tepoch'

FPATH = './results/{}.csv'.format(VAR)

XLABEL = 'neuronen per hidden layer'
YLABEL = 'nauwkeurigheid(%)'

# size of graph
YMAX    = 50
YSTEP   = 10
XMAX    = 100
XSTEP   = 20

# set font 
font = {
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

# arr to store data of file
arr   = []
# arr to store x
x_val = []
# arr to store y
y_val = []

# get data
with open(FPATH, 'r') as fo:
    for lines in fo:
        arr.append(lines.replace('\n','').split(';'))

# convert to float
for i in range(len(arr)):
    for j in range(2):
        arr[i][j] = float(arr[i][j])

# fill x_val and y_val
for i in range(len(arr)):
    x_val.append(arr[i][0])
    y_val.append(arr[i][1])

# sizes
xmin = 0
xmax = XMAX
ymin = 0
ymax = YMAX

# set sizes
plt.figure(figsize=(16,6))
plt.axis([xmin, xmax, ymin, ymax])

# # set step size
plt.xticks(np.arange(xmin, xmax+XSTEP, XSTEP))
plt.yticks(np.arange(ymin, ymax+YSTEP, YSTEP))
plt.minorticks_on()

# make grid
plt.grid(b=True, which='both', linestyle='--')
plt.axhline(y=0, xmin=xmin, xmax=xmax, color='k')
plt.axvline(x=0, ymin=ymin, ymax=ymax, color='k')

# set title and axis labels
# plt.title(TITLE)
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)

# plot points
plt.plot(x_val, y_val,'-o', ms=2)

# calc trendline
# val = np.polyfit(x_val, y_val, 6)
# lin = np.poly1d(val)
# plt.plot(x_val, lin(x_val), 'g--')
# show
plt.show()