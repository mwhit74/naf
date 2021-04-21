#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 08:43:09 2021

@author: m

Suppose that we do not know all of the three r.h.s.'s of p18 in advance.

a. Solve Ax = b1 by Gaussian elimination, getting the LU decomposition. Then,
use the LU to solve with the other two r.h.s.'s

b. Repeat part (a), this time using Crout reduction.

See below for matrix A and, b1, b2, and b3.
"""

from naf.linalg import gedo, dosv, gecr, crsv
import numpy as np

a = np.array([[4,2,1,-3],
              [1,2,-1,0],
              [3,-1,2,4],
              [0,2,4,3]],dtype=float)

b1 = np.transpose(np.array([4,2,8,9],dtype=float))

b2 = np.transpose(np.array([9,1,8,4],dtype=float))

b3 = np.transpose(np.array([4,2,-7,5],dtype=float))

lu, ov = gedo(a, pivot=False)

print(lu)

x1 = dosv(lu, ov, b1)

print(x1)

x2 = dosv(lu, ov, b2)

print(x2)

x3 = dosv(lu, ov, b3)

print(x3)

crlu, ov = gecr(a, pivot=False)

print(crlu)

x1 = crsv(crlu, ov, b1)

print(x1)

x2 = crsv(crlu, ov, b2)

print(x2)

x3 = crsv(crlu, ov, b3)

print(x3)

