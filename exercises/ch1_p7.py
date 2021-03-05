#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 07:58:57 2021

@author: m

Ch1 p7

Repeat exercise 4, this time using the method of false position. Compare the 
number of iterations required with the number used in bisection and with the 
number used in the secant method.
"""

from matplotlib import pyplot as plt
import numpy as np
import math
from naf.nonlin import li

def f1(x):
    return -math.exp(x) + 2 - math.sin(2*x)

def f2(x):
    return math.pow(x,4) - 2*x - 1

def f3(x):
    return math.cos(3*x) + 1 - math.exp(math.pow(x,2))

def f4(x):
    return -math.exp(x-1) + math.pow(x,3) + 2

# x1_range = np.linspace(-2.5, 2.5, 75)
# x2_range = np.linspace(-2.5, 2.5, 50)
# x3_range = np.linspace(-1, 1, 50)
# x4_range = np.linspace(5,7, 50)

# vf1 = np.vectorize(f1)
# vf2 = np.vectorize(f2)
# vf3 = np.vectorize(f3)
# vf4 = np.vectorize(f4)

val = li(f1, 0, 1, 0.00001)
print(val)

val = li(f2, 1, 2, 0.00001)
print(val)

val = li(f3, 0, 1, 0.00001)
print(val)

val = li(f4, 6, 7, 0.00001)
print(val)
