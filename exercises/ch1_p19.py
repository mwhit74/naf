#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 07:32:06 2021

@author: m

Ch1 p19

Beginning with the interval [0.9, 1.1], use the secant method
on the root at x = -1 in exercise 18. Can you explain why the 
secant method works better than Newton's method in this case?

"""

from naf.nonlin import secant
import math

f = lambda x: math.pow(x+1, 3)*(x-1)

s1 = secant(f, 0.9, 1.1)
print(s1)

s1 = secant(f, -0.9, -1.1)
print(s1)

"""
Discussion
---------

The secant method works better in this case than Newton's method because
Newton's method relies on the theoretical derivative of the equation. 
The deravitive function has the same challanges related to the region of 
uncertainty and thus a root can only be approximated in this region. 

However, using the secant method an approximate derivative of the function is
used. This is computed using actual values of the from the function rather
than theoretical values and can walk itself to an approximate solution. The
theroetical derivative equation is limited by the exactness of the equation
and cannot walk itself to an approximate solution. 

The theortical solution is limited because the theoretical solution tries to 
calculate an exact answer with an approximate computation machine. 

"""
