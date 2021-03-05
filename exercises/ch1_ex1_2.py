#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 07:29:21 2021

@author: m

Ch1 Ex 1.2

Find a root between 0 and 1 of the transcendental function

f(x) = 3x + sin(x) - e^x

"""

from naf.nonlin import muller
import numpy as np
import matplotlib.pyplot as plt
import cmath
import math

f = lambda x: 3*x + math.sin(x) - math.exp(x)

m1 = muller(f, 0.5, 1.0, 0.0)
print(m1)

def graph():
    vf = np.vectorize(f)
    
    xrng = np.linspace(-2,2)
    
    plt.grid()
    plt.plot(xrng, vf(xrng))
    plt.show()
    
#graph()
