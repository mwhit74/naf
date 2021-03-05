#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 08:35:48 2021

@author: m

Ch1 p12

Repeat exercise 4, but this time use Newton's method. Compare the number of 
iterations required with bisection (exercise 4), the secant method 
(exercise 6), and regula falsi (exercise 7)

"""

import matplotlib.pyplot as plt
import numpy as np
import math
from naf.nonlin import newtone

f1 = lambda x: -math.exp(x) + 2 - math.sin(2*x)
df1 = lambda x: -math.exp(x) - 2*math.cos(2*x)

f2 = lambda x: math.pow(x,4) - 2*x - 1
df2 = lambda x: 4*math.pow(x,3) - 2

f3 = lambda x: math.cos(3*x) + 1 - math.exp(math.pow(x,2))
df3 = lambda x: -3*math.sin(3*x) - 2*x*math.exp(math.pow(x,2))

f4 = lambda x: -math.exp(x-1) + math.pow(x,3) + 2
df4 = lambda x: -math.exp(x-1) + 3*math.pow(x,2)

# x1_range = np.linspace(-2.5, 2.5, 75)
# x2_range = np.linspace(-2.5, 2.5, 50)
# x3_range = np.linspace(-1, 1, 50)
# x4_range = np.linspace(5,7, 50)

# vf1 = np.vectorize(f1)
# vf2 = np.vectorize(f2)
# vf3 = np.vectorize(f3)
# vf4 = np.vectorize(f4)

# plt.plot(x1_range, vf1(x1_range))
# plt.grid(b=True, which='major', axis='both')
# plt.show()
# plt.plot(x2_range, vf2(x2_range))
# plt.grid(b=True, which='major', axis='both')
# plt.show()
# plt.plot(x3_range, vf3(x3_range))
# plt.grid(b=True, which='major', axis='both')
# plt.show()
# plt.plot(x4_range, vf4(x4_range))
# plt.grid(b=True, which='major', axis='both')
# plt.show()

n1 = newtone(0.5, f1, df1)
print(n1)

n2 = newtone(1, f2, df2)
print(n2)

n3 = newtone(0.25, f3, df3)
print(n3)

n4 = newtone(7, f4, df4)
print(n4)

"""

Number of iterations per method and function
           function
           --------
           f1    f2    f3    f4
method
------
bisect     17    17    17    17
secant     5     8     6     7
li         5     21    9     12
newton     2     5     3     3
"""
