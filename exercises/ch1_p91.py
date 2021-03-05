#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:17:30 2021

@author: m

Ch1 p91

A shpere of density d and raidus r weight 4/3*pi*r^3*d. The volumne of a
spherical segment is 1/3*pi*(3*r*h^2-h^3). Find the depth to which a sphere
of density 0.6 sinks in the water as a fraction of its radius. (See
accompanying figure.)

r - radius of sphere
d - density of sphere
h - depth of submerge sphere

***See notebook for equation derivation and assumptions

"""

import math
import matplotlib.pyplot as plt
from naf.nonlin import horner, ndpnm, quadratic_roots
import numpy as np

f = lambda x: x**3 - 3*x**2 + 2.4

p = np.array((2.4, 0, -3, 1))

#simple loop to calculate multiple roots of polynomial using
#synethic division to find the root and deflate the function
#to find the next root
for x in range(len(p)-1):
    print(p)
    p = ndpnm(p, 1)
    print(p[1])
    p = p[0][:-1]
    print(p)
    
def graph():
    xr = np.linspace(-2.5, 5)
    fig, axes = plt.subplots()
    axes.grid()
    axes.plot(xr, f(xr))
    plt.show()
    
#graph()
