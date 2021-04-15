#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:11:42 2021

@author: m

Ch2 p15

Using Gaussian elimination with partial pivoting and back-substitution

a. Solve the equations of exercise 10
b. Using part (a), find the determinant of the coefficient matrix
c. What is the LU decomposition of the coefficient matrix ?


"""

import numpy as np
from naf.linalg import gega, gesv
from naf.linalg import det

a = np.array([[1,1,-2],
              [4,-2,1],
              [3,-1,3]], dtype=float)

b = np.array([3,5,8], dtype=float)

lu, ov = gega(a)

print(lu, ov)

print(gesv(lu, ov, b))

print(det(a))