#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 08:22:52 2021

@author: m

Ch1 p18

f(x) = [(x+1)^3]*(x-1) obviously has roots at x = -1 and x = 1.
Using starting values that differ form the roots by 0.1, compare
the number of iterations taken when Newton's method computes both
of the roots until they are within 0.0001 of the correct values.
"""
import math
from naf.nonlin import newtone
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: math.pow(x+1, 3)*(x-1)

df = lambda x: 3*math.pow(x+1,2)

# vf = np.vectorize(f)

# xr = np.linspace(-1.5,1.5)

# plt.grid()
# plt.plot(xr, vf(xr))
# plt.show()

n1 = newtone(f, df, -1.1, root_tol = 0.0001, zero_tol = 0.0001)
print(n1)

n1 = newtone(f, df, -0.9, root_tol = 0.0001, zero_tol = 0.0001)
print(n1)

n1 = newtone(f, df, 0.9, root_tol = 0.0001, zero_tol = 0.0001)
print(n1)

n1 = newtone(f, df, 1.1, root_tol = 0.0001, zero_tol = 0.0001)
print(n1)


"""
Discussion
----------

Both positive starting values converge to the root at x=1. The -0.9 values
convergaes on root at x=1, and the -1.1 starting value "flies-off" to 
negative infinity. 
I find it rather unusual that the -0.9 starting value moves away from
the root at x=-1 and find the root at x=1. It is also unusual that the -1.1
value "flieis-off" to negative infinity

After re-reading the text I decided to deflate the equation by dividing by
(x-1), effectively removing the root at x=1. Then, using the negative starting
values try to solve for the x=-1 root again. See below.

This method works here because we know the roots in advanced from inspection.
If we were not able to inspect the equation for roots this would be a more
difficult problem. If the equation is a polynomial, one can use synethic
division to deflate the function as well. 
"""


g = lambda x: math.pow(x+1,3)

dg = lambda x: 3*math.pow(x+1,2)


n1 = newtone(g, dg, -1.1, root_tol = 0.0001, zero_tol = 0.0001)
print(n1)

n1 = newtone(g, dg, -0.9, root_tol = 0.0001, zero_tol = 0.0001)
print(n1)


"""
Discussion
----------

The deflated equation now converges to the x=-1 root for both negative 
starting values. Note that while the convergence of f(x) to zero is very good,
the convergence to x=-1 is not very good. The text describes how this can
happen for multiple roots and refers to it as a "neighborhood of uncertainty".
This is due to the imprecision of the computing device. 
"""
