import math
import numpy as np

#bisection algorithem
def Bisect(fnc, x1, x2, tol = 0.001, iter_limit = 50):
    """
    Root finding method using interval halving.

    The bisection method in mathematics is a root-finding method that repeatedly
    bisects an interval and then selects a subinterval in which a root must lie
    for further processing.

    Args:
        fnc (function): mathematical function to evaluate in terms of a single
            variable
        x1 (float): first guess
        x2 (float): second guess
        tol (float, optional): solution tolerance to analytically correct
            answer. Defaults to 0.001
        iter_limit (int, optional): maximum number of iterations to perform if the
            solution is not coverging. Defaults to 50.

    Returns:
        [float, float] if successful.

        The first index is the approximate root of the equation determined by
        the bisection method and the second index is the value of the function
        at the approximated root.

    Raises:
        ValueError: If the value of the mathematical function at x1 and x2 are
            of the same sign. That is if y1 = fnc(x1) and y2 = fnc(x2) are of the same
            sign the python function will raise an error and terminate.
        
    """

    iter = 1 #initalize iteration counter
    f3 = 1.0 #dummy value
    while abs(x2-x1)/2 > tol or f3 == 0 or iter > iter_limit:
        f1 = fnc(x1)
        f2 = fnc(x2)
        #if both values are positive test fails
        #if both values are negative test fails
        #if one value is positive and one negative test passes
        if f1*f2 > 0:
            raise ValueError("The values" + str(f1) + " and " + str(f2) +\
                    " do not bracket root")
            break
        x3 = (x1+x2)/2
        f3 = fnc(x3)
        print x1, x2, x3, f3
        if f3*f1 < 0:
            x2 = x3
        else:
            x1 = x3
        iter += 1

    return x3, fnc(x3)

if __name__ == "__main__":
    #test case
    def fnc(c):
        a = 123.0*math.pi/180.0
        return (-9.0*math.cos(a+c)/pow(math.sin(a+c),2)
                -7.0*math.cos(c)/pow(math.sin(c),2))

    x1 = 0.4
    x2 = 0.5
    tol = 0.0001

    print Bisect(fnc, x1, x2, tol)
