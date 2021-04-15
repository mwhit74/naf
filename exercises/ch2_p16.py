#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:30:20 2021

@author: m
"""
import numpy as np
from naf.linalg import gedo

np.set_printoptions(precision=5, suppress=True)

a = np.array([[2.51, 1.48, 4.53],
              [1.48, 0.93, -1.30],
              [2.68, 3.04, -1.48]])

lu, ov = gedo(a, pivot=False)

print(lu)

