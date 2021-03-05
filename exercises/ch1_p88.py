#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 07:55:57 2021

@author: m

Ch1 p88

In Chapter 5, a particularly efficient method for numerical integration
of a function, call Gaussian quadrature, is discussed. In the development
of formulas for this method, it is necessary to evaluate the zeros of 
Legendre polynomials. Find the zeros of the Legendre polynomial of
sixth order:

P_6(x) = 1/48 (693x^6 - 945x^4 + 315x^2 - 15)

(Note: All the zeros of Legendre polynomials are less thanone in magnitude
and, for polynomials of even order, are symmertical about the origin)

"""

import math
import matplotlib.pyplot as plt
from naf.nonlin import horner, ndpnm, quadratic_roots
import numpy as np

f = lambda x: 1/48*(693*x**6 - 945*x**4 + 315*x**2 - 15)

p = 1/48 * np.array((-15, 0, 315, 0, -945, 0, 693))

#simple loop to calculate multiple roots of polynomial using
#synethic division to find the root and deflate the function
#to find the next root
for x in range(len(p)-1):
    p = ndpnm(p, 0.5)
    print(p[1])
    p = p[0][:-1]
    
    
def graph():
    xr = np.linspace(-1, 1)
    fig, axes = plt.subplots()
    axes.grid()
    axes.plot(xr, f(xr))
    plt.show()
    
#graph()
