#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 07:33:51 2021

@author: m

Example of tri-diagonal solution from the text book, section 2.5. 

Worked example by hand and implemented algorithm in computer
program for practice, understanding, and future use. 
"""

import numpy as np
from naf.linalg import tddo, tdsv
import pdb

a = np.array([[0, -4, 2],
              [1, -4, 1],
              [1, -4, 1],
              [1, -4, 1],
              [2, -4, 0]], dtype=float)

b = np.transpose(np.array([0,-4,-11,5,6], dtype=float))

lu = tddo(a)

x = tdsv(lu, b)

print(x)
