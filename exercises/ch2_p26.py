#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 07:30:00 2021

@author: m

Solve this tri-diagonal system
...
by the special algorithm that stores the augument matrix
in a 5x4 array.

"""

import numpy as np
from naf.linalg import tddo, tdsv
import pdb

a = np.array([[0, 4, -1],
              [-1, 4, -1],
              [-1, 4, -1],
              [-1, 4, -1],
              [-1, 4, 0]], dtype=float)

b = np.transpose(np.array([100, 200, 200, 200, 100], dtype=float))

lu = tddo(a)

x = tdsv(lu, b)

print(x)
