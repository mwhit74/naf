#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 08:20:18 2021

@author: m
"""

"""Find where the cubic y = x^3 - 2x^2 + x - 1 intersects the parabola
y = 2x^2 + 3x + 1. from the graph of the two equations, locate the intersection
approximately, the use both regula falsi (linear interpolation) and the secant
method to refine the value until you are sure that you have it correct to five
sig. figs. How many iterations are needed in each case."""

from matplotlib import pyplot as plt
import numpy as np
import math
from naf.nonlin import secant, li

f1 = lambda x: math.pow(x,3) - 2*x**2 + x - 1
f2 = lambda x: 2*x**2 + 3*x + 1

f3 = lambda x: math.pow(x,3) - 4*x**2 - 2*x - 2

x1_range = np.linspace(-2, 5, 50)

vf1 = np.vectorize(f1)
vf2 = np.vectorize(f2)
vf3 = np.vectorize(f3)

plt.plot(x1_range, vf1(x1_range), x1_range, vf2(x1_range))
plt.grid()
plt.show()

plt.grid(b='Blue', which='major', axis='both')
plt.plot(x1_range, vf3(x1_range))
plt.show()

s = secant(f3, 4., 5., 0.00001)
print(s)
print(f1(s[0]))
print(f2(s[0]))

l = li(f3, 4., 5., 0.00001)
print(l)
print(f1(l[0]))
print(f2(l[0]))


