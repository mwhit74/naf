import numpy as np
import math

def stepwise_search(fnc, x1, x2, tol = 0.001, incr = 10):
    """
    Finds root by reducing the incremental step when a sign change is detected.

    The stepwise search root finding algorithm steps through the enclosing
    interval beginning at x=x1 with steps equal to (x1+x2)/incr until a sign
    change is detected. Then it steps back through the now smaller enclosing
    interval in steps 1/incr as large to more precisely isolate the root. The
    program continus the reverals of direction with smaller steps until an
    accuracy of tol is achieved.

    Arguements:
        fnc (function): mathematical function to evaluate in terms of a single
        variable
        x1 (float): first guess
        x2 (float): second guess
        tol (float, optional): solution tolerance to analytically correct answer. Defaults
        to 0.001
        incr (int, optional): number of steps to create in the interval. This value is
        also the scaler multiplier to increase the number of steps for each
        iteration. For example incr=10, so there will be 10 equal steps in the
        initial iteration and 10*10=100 equal steps in the second interation.
        Defaults to 10.

    Returns:
        [float, float] is successful.

        The first index is the approximate root of the equation determined by the
        stepwise search method and the second index is the value of the function at
        the approximated root.

    """
    yb = fnc(x2)
    prev_x_step = x1 
    prev_y_step = fnc(x1)
    step = (x1+x2)/incr
    x_step = x1 + step
    y_step = fnc(x_step)
    error = 1.0 #dummy value
    while error > tol:
        while np.sign(y_step) != np.sign(yb):
            prev_x_step = x_step
            x_step = x_step + step
            prev_y_step = y_step
            y_step = fnc(x_step)
        #reverse direction with smaller step
        error = abs(prev_y_step + y_step)/2
        est_val = (prev_x_step + x_step)/2
        incr = incr*incr
        yb = prev_y_step
        step = (x1+x2)/incr*np.sign(step)*-1
        x_step = x_step + step
        y_step = fnc(x_step)
    return est_val, fnc(est_val)


if __name__ == "__main__":

    #define interval
    x1 = 0.3
    x2 = 0.6
    incr = 10
    tol = 0.01

    #define function
    a = 2.147
    w1 = 9.0
    w2 = 7.0
    fnc = lambda c: -w1*math.cos(a+c)/pow(math.sin(a+c),2) - w2*math.cos(c)/pow(math.sin(c),2) 

    print stepwise_search(fnc, x1, x2)
