#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:09:18 2021

@author: m

ch2 p4

A = 9   2
    10 -3
    
B =  6  1  -1
     1  4   3
    -1  3  -2
    
a. Find the characteristic polynomials of both A and B
b. Find the eigenvalues of both A and B

a.
l = lambda
Matrix A: l^2 - 6l - 47

Matrix B: -l^3 + 8l^2 + 7l -110

(algebra performed by hand, see notebook)

b. 
eigenvalues computed by code below

Matrix A: [-4.48331477 10.48331514]
Matrix B: [ 6.4244289   4.99999977 -3.42444974]
"""

import numpy as np
from naf.nonlin import mrsv

pc = np.array([-47, -6, 1])
x = np.ones(pc.size - 1)

print(mrsv(pc, x))

pc = np.array([-110, 7, 8, -1])
x = np.ones(pc.size - 1)

print(mrsv(pc, x))