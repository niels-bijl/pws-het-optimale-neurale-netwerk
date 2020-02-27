#/usr/bin/env python3

'''
Plot results
@autor: Bijl, N
'''

# -> get array with all values
# -> convert to str to float
# -> get desired variable + accuracy
# -> plot values

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

DIR  = "./results"
FILE = "Result_{}.csv".format("Hidden")
#FILE = "Tanh_Hidden.csv" 

XLABEL = "neuronen per hidden layer"
YLABEL = "nauwkeurigheid(%)"
TITLE  = "{},{}".format(YLABEL, XLABEL)

XSTEP  = 50
YMAX   = 60
YSTEP  = 20

var_n  = "hidden" 

font = {
    'weight' : 'bold',
    'size'   : 18
}

matplotlib.rc('font', **font)

path = os.path.join(DIR, FILE)

var_name = {
    'hidden':   0,
    'layers':   1,
    'lr':       2,
    'act':      3,
    'epoch':    4,
    'train':    5,
    'test':     6,
    'accuracy': 7
}

nvar = var_name[var_n]

# array to store all data
arr  = []
# array to store variable
var  = []
# array to store accuracy
acc  = []

with open(path, 'r') as fo:
    for lines in fo:
        arr.append(lines.replace('\n','').split(';'))


# convert to float exceptr str
for i in range(len(arr)):
    for j in range(len(arr[i])):
        try:
            arr[i][j] = float(arr[i][j])
        except ValueError:
            pass



# remove header and empty
for i in range(len(arr)):
    if(all(type(arr[i][j]) is str for j in range(len(arr[i]))) or not arr[i]):
        arr[i].clear()

arr = [x for x in arr if x !=[]]

# get desired var and acc        
for i in range(len(arr)): 
    var.append(arr[i][nvar])
    acc.append(arr[i][var_name['accuracy']])

print(var,acc)
# sizes
xmin = 0
xmax = max(var)
ymin = 0
ymax = YMAX

# set sizes
plt.figure(figsize=(16,6))
plt.axis([xmin, xmax, ymin, ymax])

# set step size
plt.xticks(np.arange(xmin, xmax+XSTEP, XSTEP))
plt.yticks(np.arange(ymin, ymax+YSTEP, YSTEP))
plt.minorticks_on()

# make grid
plt.grid(b=True, which='both', linestyle='--')
plt.axhline(y=0, xmin=xmin, xmax=xmax, color='k')
plt.axvline(x=0, ymin=ymin, ymax=ymax, color='k')

# set title and axis labels
plt.title(TITLE)
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)

# plot points
plt.plot(var, acc,'.', ms=4)

# calc trendline
val = np.polyfit(var, acc, 6)
lin = np.poly1d(val)
plt.plot(var, lin(var), 'g--')
# show
plt.show()