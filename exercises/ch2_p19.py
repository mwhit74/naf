#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 07:35:26 2021

@author: m
"""

from sympy import symbols, simplify, factor

n = symbols('n')

f1 = n*(n-1)/2
f2 = n*(2*n-1)*(n-1)/6
f3 = f1

f4 = simplify(f1 + f2 + f3)
print(f4)