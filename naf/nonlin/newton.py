"""Newton's Method - mk.1

Experimenting with algorithm to solve square roots.
"""

def f(x, c):
    return x**2-c

def df(x):
    return 2*x

c = 101.0
x0 = xp = c/2.0

e = 1.0
count = 0

while e > 0.000000001:
    xn = xp - f(xp, c)/df(xp)
    e = f(xn, c)
    xp = xn
    count += 1

print(xp, count)
     
