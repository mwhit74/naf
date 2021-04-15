#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:54:59 2021

@author: m

Ch2 p11

Show that the following does not have a solution

3  2 -1 -4 | 10
1 -1  3 -1 | -4
2  1 -3  0 | 16
0 -1  8 -5 |  3

p12

If the r.h.s of p11 is transpose([2,3,1,3]) show that there
are an infinite number of solutions.
"""

import numpy as np
from naf.linalg import gega, gesv

a = np.array([[3,2,-1,-4],
              [1,-1,3,-1],
              [2,1,-3,0],
              [0,-1,8,-5]],dtype=float)

b1 = np.array([10,-4,16,3], dtype=float)
b2 = np.array([2,3,1,3], dtype=float)

lu, ov = gega(a)

print(lu, ov)

print(gesv(lu, ov, b1))

print(gesv(lu, ov, b2))

