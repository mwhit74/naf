#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 08:15:37 2021

@author: m

Find the max/min points of the function

f(x) = [sin(x)]^6 * e^(20x) * tan(1-x)

on the interval [0,1]. Compare your own root-finding program to 
the IMSL subroutine ZBRENT. [Note the disdavantage in trying to
solve f'(x) = 0 using Newton's method.]

***See notebook for additional discussion.

"""

import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
from sympy import symbols, diff, exp, sin, tan
from naf.nonlin import muller, muller_c, fpi, secant, bisect

f = lambda x: pow(math.sin(x),6)*math.exp(20*x)*math.tan(1-x)

#x = symbols('x')

#print(diff(pow(sin(x),6)*exp(20*x)*tan(1-x)))

df = lambda x: (-(math.tan(x - 1)**2 + 1)*math.exp(20*x)*pow(math.sin(x),6)
                - 20*math.exp(20*x)*pow(math.sin(x),6)*math.tan(x - 1) 
                - 6*math.exp(20*x)*pow(math.sin(x),5)*math.cos(x)*math.tan(x - 1))

# cdf = lambda x: (-(cmath.tan(x - 1)**2 + 1)*cmath.exp(20*x)*pow(cmath.sin(x),6)
#                 - 20*cmath.exp(20*x)*pow(cmath.sin(x),6)*cmath.tan(x - 1) 
#                 - 6*cmath.exp(20*x)*pow(cmath.sin(x),5)*cmath.cos(x)*cmath.tan(x - 1))

def graph():
    vf = np.vectorize(f)
    vdf = np.vectorize(df)
    xr1 = np.linspace(0, 0.2)
    xr2 = np.linspace(0.2, 0.4)
    xr3 = np.linspace(0.4, 0.6)
    xr4 = np.linspace(0.6, 0.8)
    xr5 = np.linspace(0.8, 1.0)
    xr6 = np.linspace(0, 1,1000)
    
    for xr in (xr1, xr2, xr3, xr4, xr5, xr6):
        fig, axes = plt.subplots()
        axes.grid()
        axes.plot(xr, vf(xr))
        axes.plot(xr, vdf(xr))
        plt.show()
    
graph()

print('\nFirst Root\n')

b1 = bisect(df, -0.4, 0.5)
print(b1)

s1 = secant(df, 0.1, 0.25)
print(s1)

fp1 = fpi(df, 0.1)
print(fp1)

m1 = muller(df, 0.05, 0.25, 0.5)
print(m1)

print('\nSecond Root\n')
b1 = bisect(df, 0.8, 1.0)
print(b1)

#tried secant, fixed-point iteration, and muller
#none of them would converge to the root near 0.95
