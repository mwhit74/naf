#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 07:57:52 2021

@author: m

Ch1 p11

Find the root near x = 1 of y = exp(x-1) - 5x^3, beginning with x = 1. How 
accurate is the estimate after four iterations of Newton's method? How many 
iterations of bisection does it take to achieve the same accuracy? Tabulate 
the number of correct digits at each iteration of Newton's method and see 
if these double each time.
"""

import matplotlib.pyplot as plt
import math
import numpy as np
from naf.nonlin import newtone, bisect

f1 = lambda x: math.exp(x-1) - 5*math.pow(x,3)
df1 = lambda x: math.exp(x-1) - 15*math.pow(x,2)

# vf1 = np.vectorize(f1)
# vdf1 = np.vectorize(df1)

# x1_range = np.linspace(-1, 1)

# plt.grid()
# plt.plot(x1_range, vf1(x1_range))
# plt.show()

n = newtone(1, f1, df1, root_tol = 0.000001, zero_tol = 0.000001, iter_limit = 4, verbose=True)
print(n)
b = bisect(f1, -0.5, 1, root_tol = 0.000001, zero_tol = 0.000001, iter_limit=13, verbose=True)
print(b)
