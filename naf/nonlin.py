
import math
import cmath
import numpy as np


def stepwise_search(fnc, x1, x2, tol = 0.0001, incr = 10):
    
    """
    Finds root by reducing the incremental step when a sign change is detected.

    The stepwise search root finding algorithm steps through the enclosing
    interval beginning at x=x1 with steps equal to (x1+x2)/incr until a sign
    change is detected. Then it steps back through the now smaller enclosing
    interval in steps 1/incr as large to more precisely isolate the root. The
    program continus the reverals of direction with smaller steps until an
    accuracy of tol is achieved.

    Parameters
    ----------
        fnc : function
            mathematical function to evaluate in terms of a single variable
        x1 : float
            initial estimate of root #1
        x2 : float
            initial estimate of root #2
        tol : float, optional
            solution tolerance to analytically correct answer. 
            The default is 0.0001.
        incr : int, optional
            number of steps to create in the interval. This value is
            also the scaler multiplier to increase the number of steps for each
            iteration. For example incr=10, so there will be 10 equal steps in 
            the initial iteration and 10*10=100 equal steps in the second 
            interation.
            The default is 10.

    Returns
    -------
        Tuple(x2, y2)
        
        x2 : float
            approximate value of the root
        y2 : float
            approximate value of the function at the root

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






#bisection algorithm
def bisect(fnc, x0, x1, root_tol = 0.0001, zero_tol = 0.0001, iter_limit = 50, 
           verbose=False):
    """
    Root finding method using interval halving.

    The bisection method in mathematics is a root-finding method that repeatedly
    bisects an interval and then selects a subinterval in which a root must lie
    for further processing.

    Paremeters
    ----------
        fnc : function
            mathematical function to evaluate in terms of a single variable
        x0 : float
            initial estimate of root #1
        x1 : float
            initial estimate of root #2
        root_tol : float, optional
            soluation tolerance of root value. The default is 0.0001.
        zero_tol : float, optional
            solution tolerance of function value. The default is 0.0001.
        iter_limit : int, optional 
            maximum number of iterations to perform if the solution is not 
            coverging. Defaults to 50.
        verbose : bool, optional
            will print num_iter, x1, x2, x3, and f(x3) at each iteration step

    Returns
    -------
        Tuple(x2, y2, num_iter) if successful.
        
        x2 : float
            approximate value of the root
        y2 : float
            approximate value of the function at the root
        num_iter : int
            number of iterations to find approximate root

    Raises
    ------
        ValueError : 
            If the value of the mathematical function at x1 and x2 are
            of the same sign. That is if y1 = fnc(x1) and y2 = fnc(x2) are of 
            the same sign the python function will raise an error and 
            terminate.
    """

    num_iter = 0 #initalize iteration counter
    y2 = fnc(x1+x0)/2
    x2 = (x1+x0)/2
    
    while (not(abs(x1-x0)/2 < root_tol or abs(y2) < zero_tol) 
            and num_iter < iter_limit):
        y0 = fnc(x0)
        y1 = fnc(x1)
        #if both values are positive test fails
        #if both values are negative test fails
        #if one value is positive and one negative test passes
        if y0*y1 > 0:
            raise ValueError(f'The values {y0} and {y1} do not bracket root')
            break
        x2 = (x0+x1)/2
        y2 = fnc(x2)

        if y2*y0 < 0:
            x1 = x2
        else:
            x0 = x2
        
        num_iter += 1    
        
        if verbose:
            print(num_iter,x0, x1, x2, y2)
            
        

    return x2, y2, num_iter






def secant(fnc, x0, x1, zero_tol=0.0001, iter_limit=50, verbose=False):
    """
    Root finding method using the secant method.
    
    The secant method linearly extrapolates the root of the function with 
    successive iterations. Geometrically, it draws a straight line through
    two known points of the function and extends it through the x-axis. This 
    x-value gives the next point on the function and the closest of the two
    previous points is used to find the next approximate root in the same
    fashion. 

    Parameters
    ----------
        fnc : function
            mathematical function to evaluate in terms of a single variable
        x0 : float
            initial estimate of root #1
        x1 : float
            initial estimate of root #2
        tol : float, optional
            solution tolerance to analytically correct answer. 
            The default is 0.0001.
        iter_limit : int, optional
            maximum number of iterations to perform if the solution is not 
            coverging. The default is 50.
        verbose : bool, optional
            will print num_iter, x0, x1, x2, and f(x2) at each iteration step. 
            The default is False.

    Returns
    -------
        Tuple(x2, y2, num_iter)
        
        x2 : float
            approximate value of the root
        y2 : float
            approximate value of the function at the root
        num_iter : int
            number of iterations to find approximate root

    """
    
    num_iter = 0
    x2 = zero_tol*10000
    
    if fnc(x0) < fnc(x1):
        x_temp = x0
        x0 = x1
        x1 = x_temp
    
    while abs(fnc(x2)) > zero_tol and num_iter < iter_limit:
           
        x2 = x1 - fnc(x1)*(x0-x1)/(fnc(x0)-fnc(x1))
        x0 = x1
        x1 = x2
        
        num_iter += 1
        
        if verbose:
            print(num_iter, x0, x1, fnc(x1))
        
        
        
    return (x2, fnc(x2), num_iter)






def li(fnc, x0, x1, tol = 0.0001, iter_limit = 50, verbose=False):
    """
    Root finding method using linear interpolation (false position).

    The linear interpolation method ensures the existance of root by approaching
    it from both sides, i.e. above and below the x-axis. The algorithem keeps
    the root bracketed for each successive iteration. 

    Parameters
    ----------
        fnc : function
            mathematical function to solve in terms of a single variable
        x0 : float 
            initial estimate of root #1* (first guess)
        x1 : float
            initial estimate of root #2* (second guess)
        tol : float, optional
            solution toleration to analytically correct answer. 
            Defaults to 0.0001
        iter_limit : int, optional
            maximum number of iterations to persom if the solution is not 
            converging. Defaults to 20. 
        verbose : bool, optional
            will print num_iter, x1, x2, x3, and f(x3) at each iteration step. 
            The default is False.

        *Note: The values x0 and x1 must bracket the root. 
        
    Returns
    -------
        Tuple(x2, y2, num_iter)
        
        x2 : float
            approximate value of the root
        y2 : float
            approximate value of the function at the root
        num_iter : int
            number of iterations to find approximate root

    Raises:
    -------

    """

    num_iter = 0 #initialize iteration counter

    x2 = 1.0 #dummy value

    while abs(fnc(x2)) > tol and num_iter < iter_limit:
        y0 = fnc(x0)
        y1 = fnc(x1)
        
        if y0*y1 > 0:
            raise ValueError("The values" + str(y0) + " and " + str(y1) +\
                    " do not bracket root")
            break

        x2 = x1 - y1*(x0-x1)/(y0-y1)
        y2 = fnc(x2)

        #if f2 is on the opposite side of the x-axis as f0
        #x1 is updated to equal x2 since x2 is close to the root;
        #else f2 is on the same side of the x-axis as f0
        #x0 is updated to equal x2
        if np.sign(y2) != np.sign(y0):
            x1 = x2
        else:
            x0 = x2
            
        num_iter += 1
            
        if verbose:
            print(num_iter, x0, x1, y2)

    return x2, y2, num_iter






def newtona():
    """Newton's method using finite differences to approximate derivative.
    
    
    See secant method and linear interpolation for an approximate Newton
    method. 
    """
    pass






def newtone(fnc, dfnc, x0, root_tol=0.0001, zero_tol=0.0001, iter_limit=20, 
            verbose=False):
    """Newton's method using exact derivative provided by user.
    
    Parameters
    ----------
        x0 : float
            initial estimate of root
        fnc : function
            mathematical function to solve in terms of a single variable
        dfnc : function
            derivative of function
        root_tol : float, optional
            soluation tolerance of root value. The default is 0.0001.
        zero_tol : float, optional
            solution tolerance of function value. The default is 0.0001.
        max_iter : int, optional
            maximum number of iterations to perform if the solution is not 
            converging. The default is 20.
        verbose : bool, optional
            will print num_iter, x0, x1, fnc(x1) at each iteration step. 
            The default is False.
            

    Returns
    -------
        Tuple(x1, y1, num_iter)
        
        x1 : float
            approximate value of the root
        y1: float
            value of the function at the approximated root
        num_iter : int
            number of iterations to find approximate root

    """
    

    num_iter = 0
    x1 = x0*10
    
    if fnc(x0) != 0 and dfnc(x0) != 0:
        while (not(abs(x0-x1) < root_tol or abs(fnc(x0)) < zero_tol)
                   and num_iter < iter_limit):
            x1 = x0
            x0 = x0 - fnc(x0)/dfnc(x0)
            
            num_iter += 1
            
            
            if verbose:
                print(num_iter, x1, x0, fnc(x0))
            
    return (x0, fnc(x0), num_iter)







def muller(fnc, x0, x1, x2, zero_tol = 0.0001, iter_max = 20,
            verbose = False):
    """Muller's method

    Parameters
    ----------
        fnc : function
            mathematical function to solve in terms of a single variable
        x0 : flaot
            initial estimate of root #1
        x1 : float
            initial estimate of root #2
        x2 : float
            initial estimate of root #3
        zero_tol : float, optional
            solution tolerance of function value. The default is 0.0001.
        iter_max : float, optional
            maximum number of iterations if the solution is not converging.
            The default value is 20.
        verbose : bool, optional
            will print num_iter, x0, x1, x2, y0, y1, y2, a, b, c, xr, fnc(xr) 
            at each iteration step. 
            The default is False.

    Returns
    -------
        Tuple(xr, yr, num_iter)
        
        xr : float
            approximate value of the root
        yr: float
            value of the function at the approximated root
        num_iter : int
            number of iterations to find approximate root

    """
    num_iter = 0
    xr = zero_tol * 1000
    
    y0 = fnc(x0)
    y1 = fnc(x1)
    y2 = fnc(x2)
    
    zero_error = []
    
    while abs(fnc(xr)) > zero_tol and num_iter < iter_max:
        
        h1 = x1 - x0
        h2 = x0 - x2
        
        gamma = h2/h1
        
        a = (gamma*y1 - y0*(1+gamma) + y2)/(gamma*pow(h1,2)*(1+gamma))
        b = (y1 - y0 - a*pow(h1,2))/h1
        c = y0
        
        #chooses root nearest the middle point x0
        if b > 0:
            d = +1*math.sqrt(pow(b,2)-4*a*c)
        if b < 0:
            d = -1*math.sqrt(pow(b,2)-4*a*c)
        if b == 0:
            d = +1*math.sqrt(pow(b,2)-4*a*c)
            
        
        xr = x0 - (2*c)/(b + d)
        
        num_iter += 1
        
        if verbose:
            print(num_iter, x0, x1, x2, y0, y1, y2, xr, fnc(xr))            
        
        if xr > x0:
            x2 = x0
            x0 = xr
            x1 = x1
            
            y2 = y0
            y0 = fnc(xr)
            y1 = y1
        else:
            x2 = x2
            x1 = x0
            x0 = xr
            
            y2 = y2
            y1 = y0
            y0 = fnc(xr)
            
    return (xr, fnc(xr), num_iter)








def muller_c(fnc, x0, x1, x2, zero_tol = 0.0001, iter_max = 20,
            verbose = False):
    """
    Muller algorithm with complex numbers

    Parameters
    ----------
        fnc : function
            mathematical function to solve in terms of a single variable
        x0 : flaot
            initial estimate of root #1
        x1 : float
            initial estimate of root #2
        x2 : float
            initial estimate of root #3
        zero_tol : float, optional
            solution tolerance of function value. The default is 0.0001.
        iter_max : float, optional
            maximum number of iterations if the solution is not converging.
            The default value is 20.
        verbose : bool, optional
            will print num_iter, x0, x1, x2, y0, y1, y2, a, b, c, xr, fnc(xr) 
            at each iteration step. 
            The default is False.

    Returns
    -------
        Tuple(xr, yr, num_iter)
        
        xr : float
            approximate value of the root
        yr: float
            value of the function at the approximated root
        num_iter : int
            number of iterations to find approximate root

    """
    num_iter = 0
    xr = zero_tol * 1000
    
    y0 = fnc(x0)
    y1 = fnc(x1)
    y2 = fnc(x2)
    
    while abs(fnc(xr)) > zero_tol and num_iter < iter_max:
        
        h1 = x1 - x0
        h2 = x0 - x2
        
        gamma = h2/h1
        
        a = (gamma*y1 - y0*(1+gamma) + y2)/(gamma*pow(h1,2)*(1+gamma))
        b = (y1 - y0 - a*pow(h1,2))/h1
        c = y0
        
        #chooses root nearest the middle point x0
        # if b > 0 +0j:
        #     d = +1*cmath.sqrt(pow(b,2)-4*a*c)
        # if b < 0 +0j:
        #     d = -1*cmath.sqrt(pow(b,2)-4*a*c)
        # if b == 0 +0j:
        #     d = +1*cmath.sqrt(pow(b,2)-4*a*c)
            
        d = cmath.sqrt(pow(b,2)-4*a*c)
        e = (2*c)/(b + d)
        f = (2*c)/(b - d)
        
        if e >= f:
            xr1 = x0 - e
        else:
            xr1 = x0 - f
        
        num_iter += 1
        
        if verbose:
            print(num_iter, x0, x1, x2, y0, y1, y2, xr, fnc(xr))            
        
        if xr > x0:
            x2 = x0
            x0 = xr
            x1 = x1
            
            y2 = y0
            y0 = fnc(xr)
            y1 = y1
        else:
            x2 = x2
            x1 = x0
            x0 = xr
            
            y2 = y2
            y1 = y0
            y0 = fnc(xr)
            
    return (xr, fnc(xr), num_iter)








def fpi(fnc, x0, root_tol=0.0001, iter_max=50, verbose=False):
    """
    Fixed-Point Iteration method
    
    Root finding method that uses the current point to calculate the next point
    iteratively towards the approximated root. 

    Parameters
    ----------
    fnc : function
        mathematical function to solve in terms of a single variable;
        rearranged function in the form x = g(x)
            f(x) = x^2 - 2x - 3 = 0
            x = g(x) = sqrt(2x + 3)
    x0 : float
        initial estimate of root
    root_tol : float, optional
        Solution convergence tolerance. The default is 0.0001.
    iter_max : float, optional
        Maximum number of iterations if the solution is not coverging. 
        The default is 50.
    verbose : bool, optional
        will print num_iter, x0, x1, fnc(x1). The default is False.

    Returns
    -------
    Tuple(x1, y1, num_iter)
    
    x1 : float
        approximated root value
    y1 : float
        value of the function at the approximated root
    num_iter : int
        number of iterations to find approximate root

    """
    num_iter = 0
    x1 = root_tol * 100
    
    while abs(x0 - x1) > root_tol and num_iter < iter_max:
        x1 = x0
        x0 = fnc(x0)
        
        num_iter += 1
        
        if verbose:
            print(num_iter, x1, x0, fnc(x0))
            
    return (x0, fnc(x0), num_iter)








def horner(pc, x):
    """
    Horner's Method or synethic division.
    
    Applies synethic division to polynomial equation and returns coefficients
    plus the remainder. 
    
    This is also known as deflation of the function.

    Parameters
    ----------
    pc : list
         polynomial coefficients in order of increasing exponent power
         a_1*x^4 + a_2*x^3 + a_3*x^2 + a_4*x + a_5
         pc = [a_5, a_4, a_3, a_2, a_1]
    x : float
        initial estimate of root 

    Returns
    -------
    dpc : list
         polynomical coefficients of defleated polynomial in order of increasing
         power
    R : float
        remainder after synethic division

    """

    n = len(pc)-1
    dpc = np.zeros(n+1)
    
    for i in range(n-1, -1, -1):
        dpc[i] = pc[i+1] + x*dpc[i+1]
        
    R = pc[0] + x*dpc[0]
        
    return dpc, R







def ndpnm(pc, x, abs_error=0.0001, max_iter=20):
    """
    N-degree polynomial Newton's method
    
    This method iteratively solves for a root of an n-degree polynomial. 
    It returns the root and the coefficients of the defleated polynomial 
    equation of degree n-1. The defleated polynomial can again be solved 
    while recognizing there is a discontinuity at each root previously 
    determined by this method. 

    Parameters
    ----------
    pc : list
         polynomial coefficients in order of increasing exponent power
         
         a_1*x^4 + a_2*x^3 + a_3*x^2 + a_4*x + a_5
         
         pc = [a_5, a_4, a_3, a_2, a_1]
    x : float
        initial estimate of root 
    abs_error : float
        absolute error, convergence tolerance for root. The default is 0.0001.
    max_iter : int
        maximum number of iterations allowed to achieve convergence tolerance.
        The default is 20.

    Returns
    -------
    dpc : list
          polynomial coefficients of defleated polynomial in order of increasing
          power
    cur_x : float
        a root of the polynomial equation
        
    Usage Notes
    -----------
    #a0, a(0+1), a(0+2), ..., a(n)
    #3rd degree polynomial
    pc = np.array([110, -7, -8, 1]) #coefficients in reverse order
    guess_x = 4
    max_iter = 20
    
    dpc, x = ndpnm(f, guess_x, 0.0001, 20)
    x1, x2 = quadratic_roots(g)
    
    print(x, x1, x2)

    """
    
    count = 0
    error = abs_error * 100
    
    while error > abs_error and count < max_iter:
        
        dpc, R1 = horner(pc, x)
        tpc, R2 = horner(dpc, x)
        
        cur_x = x - R1/R2
        
        error = abs(x - cur_x)
        count += 1
        
        x = cur_x
    
    return dpc, cur_x







def quadratic_roots(pc):
    """Compute the roots of a quadratic equation.
    
    Parameters
    ----------
        pc : list
             [c, b, a] coefficients of the quadratic equation in reverse order
            
    Returns
    -------
        Tuple(x1, x2)
        
        x1 : float
            first root
        x2 : float
            second root   
    """
    a = pc[2]
    b = pc[1]
    c = pc[0]
    
    x1 = (-b + math.sqrt(b**2-4*a*c))/(2*a)
    x2 = (-b - math.sqrt(b**2-4*a*c))/(2*a)

    return x1, x2
        





def dekker(fnc, x0, x1, root_tol = 0.0001, max_iter = 50):
    pass





def brent(fnc, x0, x1, root_tol = 0.0001, max_iter = 50):
    pass
    