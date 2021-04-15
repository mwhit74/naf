#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 07:34:07 2021

@author: m


"""

import numpy as np
from naf.linalg import gega, gesv

a = np.array([[2,4,-1,-2],
              [4,0,2,1],
              [1,3,-2,0],
              [3,2,0,5]], dtype=float)

b = np.array([10,7,3,2], dtype=float)

lu, ov = gega(a)

print(lu, ov)

print(gesv(lu, ov, b))

"""
[[ 0.5     4.     -2.     -2.5   ]
 [ 4.      0.      2.      1.    ]
 [ 0.25    0.75   -1.      1.625 ]
 [ 0.75    0.5     0.5     4.6875]] 


[1 0 2 3]


[[ 2.]
 [ 1.]
 [ 2.]
 [-1.]]
"""




