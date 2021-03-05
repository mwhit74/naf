#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 08:11:45 2021

@author: m

Ch1 p22

Each of the following has comples roots. Apply Newton's
method to find them

x^2 = -2
x^3 - x^2 - 1 = 0
x^4 -3x^2 + x + 1 = 0
x^2 = (e^(-2x) - 1)/x

"""

from naf.nonlin import newtone
import cmath
import numpy as np
import matplotlib.pyplot as plt

# 2, 0 pos-real
# 0 neg-real
# 2, 0 img
f1 = lambda x: pow(x,2) + 2
df1 = lambda x: 2*x

# 1 pos-real
# 0 neg-real
# 2 img
f2 = lambda x: pow(x,3) - pow(x,2) -1
df2 = lambda x: 3*pow(x,2) - 2*x

# 2, 0 pos-real
# 2, 0 neg-real
# 4, 2, 0 img
f3 = lambda x: pow(x,4) - 3*pow(x,2) + x + 1
df3 = lambda x: 4*pow(x,3) - 6*x + 1

# 2, 0 pos-real
# 0 neg-real
# 2,0 img
f4 = lambda x: (cmath.exp(-2*x)-1)/x - pow(x,2)
df4 = lambda x: -2*x - 2*cmath.exp(-2*x)/x - (-1 + cmath.exp(-2*x))/x**2

c1 = 1 +1j
c2 = 1 -1j

c3 = 10 +5j
c4 = 10 -5j

c5 = c1
c6 = -1 -1j
c7 = c3
c8 = -10 -5j

print('Eq. 1 roots')
n1 = newtone(f1, df1, c1)
print(n1)

n1 = newtone(f1, df1, c2)
print(n1)
print('\n')

print('Eq. 2 roots')
n1 = newtone(f2, df2, c5)
print(n1)

n1 = newtone(f2, df2, c2)
print(n1)

n1 = newtone(f2, df2, c7)
print(n1)

print('\n')

print('Eq. 3 roots')
n1 = newtone(f3, df3, c5)
print(n1)

n1 = newtone(f3, df3, c6)
print(n1)

n1 = newtone(f3, df3, c7)
print(n1)

n1 = newtone(f3, df3, c8)
print(n1)

print('\n')

print('Eq. 4 roots')
n1 = newtone(f4, df4, c1)
print(n1)

n1 = newtone(f4, df4, c2)
print(n1)
print('\n')


def graph():
    vf1 = np.vectorize(f1)
    vf2 = np.vectorize(f2)
    vf3 = np.vectorize(f3)
    vf4 = np.vectorize(f4)
    
    xr = np.linspace(-10,10)
    xr3b = np.linspace(0.5, 1.5)
    xr3a = np.linspace(-2.5, 2.5)
    xr4 = np.linspace(-3,5)
    
    plt.grid()
    plt.plot(xr, vf1(xr))
    plt.show()
    
    plt.grid()
    plt.plot(xr, vf2(xr))
    plt.show()
    
    plt.grid()
    plt.plot(xr3a, vf2(xr3a))
    plt.show()
    
    plt.grid()
    plt.plot(xr3a, vf3(xr3a))
    plt.show()
    
    plt.grid()
    plt.plot(xr3b, vf3(xr3b))
    plt.show()
    
    plt.grid()
    plt.plot(xr4, vf4(xr4))
    plt.show()
    
#graph()
