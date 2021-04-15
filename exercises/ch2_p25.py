#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 08:43:09 2021

@author: m
"""

from naf.linalg import gedo, gesv, gecr, gebs, gefe
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

x1 = gesv(lu, ov, b1)

print(x1)

# x2 = gesv(lu, ov, b2)

# print(x2)

# x3 = gesv(lu, ov, b3)

# print(x3)

crlu, ov = gecr(a, pivot=False)

print(lu)

print(x1)

# x2 = gesv(lu, ov, b2)

# print(x2)

# x3 = gesv(lu, ov, b3)

# print(x3)

