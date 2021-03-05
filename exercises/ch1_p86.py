#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 07:19:38 2021

@author: m

The solution of boundary value problems by an analytical (Fourier series)
method often involves finding the roots of transcendental equations to 
evaluate the coefficients. For example:

y'' + ly = 0
y(0) = 0
y(1) = y'(1)

l - lambda

involves solving tan(z) = z. Find three values of z other than z = 0


***See notebook for additional discussion related to this problem
***
Notebook summary: tried fpi, muller, secant, and newton but none
would converge due to the very large slope of the function near the
root. Small variations in x produce large variations in y. Bisection
was the only method discussed in the text that would converge. 
***

"""

import matplotlib.pyplot as plt
import numpy as np
import math
from naf.nonlin import fpi, muller, secant, bisect, newtone

f = lambda z: math.tan(z) - z
f2 = lambda z: math.tan(z)
f3 = lambda z: math.atan(z)

df = lambda z: pow(math.tan(z), 2)

def graph():
    vf = np.vectorize(f)
    x = np.linspace(1.25, 1.75, num=100)
    
    fig, axes = plt.subplots()
    axes.grid()
    axes.plot(x, vf(x))
    plt.show()
    
#graph()
b1 = bisect(f, -0.5, 0.4) 
print(b1)

b1 = bisect(f, 1.4, 1.7)
print(b1)

b1 = bisect(f, 1.6, 4.5)
print(b1)

b1 = bisect(f, -0.5, -1.6)
print(b1)
