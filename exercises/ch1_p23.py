#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 07:51:30 2021

@author: m

Ch1 p23

Use Muller's method to solve the follwing equations.
Apply five iterations and determine how the successive
errors are related. Use starting values differ by 0.2
from each other.

3x^3 + 2x^2 - x - 6 = 0, root near 1.0
e^x -2x^3 = 0, root near 6; what are the other roots?
e^x = 3x^2, root near 3.7
tan(x) - x + 2 = 0, root near 4.3

"""

from naf.nonlin import muller
import numpy as np
import math
import matplotlib.pyplot as plt

f1 = lambda x: 3*pow(x,3) + 2*pow(x,2) - x - 6
f2 = lambda x: math.exp(x) - 2*pow(x,3)
f3 = lambda x: math.exp(x) - 3*pow(x,2)
f4 = lambda x: math.tan(x) - x + 2

m1 = muller(f1, 0.8, 1.0, 1.2, zero_tol = 0.00000000000000001, iter_max=5)
print(m1[3])
e1 = np.log(np.abs(m1[3]))
print(e1)
e2 = e1[1:-2] - e1[-1]
e1 = e1[:-3] - e1[-1]
print(e2, e1)

# m1 = muller(f2, 5.8, 6.0, 6.2, zero_tol = 0.000000000000001, iter_max=5, verbose = True)
# print(m1)

# m1 = muller(f3, 3.5, 3.7, 3.9, zero_tol = 0.000000000000001, iter_max=5, verbose = True)
# print(m1)

# m1 = muller(f4, 4.1, 4.4, 4.5, zero_tol = 0.0000000001, iter_max=5, verbose = True)
# print(m1)

def graph():
    
    y1 = lambda x: 0.1*x
    y2 = lambda x: 0.1*pow(x,2)
    y3 = lambda x: 0.1*pow(x,3)
    xr = np.linspace(20,50)
    
    fig, axes = plt.subplots()
    axes.grid()
    axes.set_yscale('log')
    axes.set_xscale('log')
    #axes.plot(x, y)
    axes.plot(xr, y1(xr))
    axes.plot(xr, y2(xr))
    axes.plot(xr, y3(xr))
    axes.plot(e1, e2, 'r--')
    plt.show()
    
graph()
