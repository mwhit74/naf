
import math
import numpy as np





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






#bisection algorithm
def bisect(fnc, x0, x1, tol = 0.001, iter_limit = 50, verbose=False):
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
            initial estimate #1
        x1 : float
            initial estimate #2
        tol : float, optional
            solution tolerance to analytically correct answer. 
            Defaults to 0.001
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

    num_iter = 1 #initalize iteration counter
    f2 = 1.0 #dummy value
    while abs(x1-x0)/2 > tol or f2 == 0 or num_iter > iter_limit:
        f0 = fnc(x0)
        f1 = fnc(x1)
        #if both values are positive test fails
        #if both values are negative test fails
        #if one value is positive and one negative test passes
        if f0*f1 > 0:
            raise ValueError("The values" + str(f0) + " and " + str(f1) +\
                    " do not bracket root")
            break
        x2 = (x0+x1)/2
        f2 = fnc(x2)

        if f2*f0 < 0:
            x1 = x2
        else:
            x0 = x2
            
        if verbose:
            print(num_iter,x0, x1, x2, f2)
            
        num_iter += 1

    return x2, fnc(x2), num_iter






def secant(fnc, x0, x1, tol=0.001, iter_limit=50, verbose=False):
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
            initial estimate #1
        x2 : float
            initial estimate #2
        tol : float, optional
            solution tolerance to analytically correct answer. 
            The default is 0.001.
        iter_limit : int, optional
            maximum number of iterations to perform if the solution is not 
            coverging. The default is 50.
        verbose : bool, optional
            will print num_iter, x1, x2, x3, and f(x3) at each iteration step. 
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
    
    num_iter = 1
    x2 = 1.0
    
    if fnc(x0) < fnc(x1):
        x_temp = x0
        x0 = x1
        x1 = x_temp
    
    while abs(fnc(x2)) > tol and num_iter < iter_limit:
           
        x2 = x1 - fnc(x1)*(x0-x1)/(fnc(x0)-fnc(x1))
        x0 = x1
        x1 = x2
        
        if verbose:
            print(num_iter, x0, x1, fnc(x1))
        
        num_iter += 1
        
    return (x2, fnc(x2), num_iter)





def li(fnc, x0, x1, tol = 0.001, iter_limit = 50, verbose=False):
    """
    Root finding method using linear interpolation (false position).

    The linear interpolation method ensures the existance of root by approaching
    it from both sides, i.e. above and below the x-axis. The algorithem keeps
    the root bracketed for each successive iteration. 

    Parameters
    ----------
        fnc : function
            mathematical function to evalue in terms of a single variable
        x0 : float 
            initial estimate #1* (first guess)
        x1 : float
            initial estimate #2* (second guess)
        tol : float, optional
            solution toleration to analytically correct answer. 
            Defaults to 0.001
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

    num_iter = 1 #initialize iteration counter

    x2 = 1.0 #dummy value

    while abs(fnc(x2)) > tol and num_iter < iter_limit:
        f0 = fnc(x0)
        f1 = fnc(x1)
        
        if f0*f1 > 0:
            raise ValueError("The values" + str(f0) + " and " + str(f1) +\
                    " do not bracket root")
            break

        x2 = x1 - f1*(x0-x1)/(f0-f1)
        f2 = fnc(x2)

        #if f2 is on the opposite side of the x-axis as f0
        #x1 is updated to equal x2 since x2 is close to the root;
        #else f2 is on the same side of the x-axis as f0
        #x0 is updated to equal x2
        if np.sign(f2) != np.sign(f0):
            x1 = x2
        else:
            x0 = x2
            
        if verbose:
            print(num_iter, x0, x1, fnc(x1))
            
        num_iter += 1

    return x2, fnc(x2), num_iter




def newton_a():
    """Newton's method using finite differences to approximate derivative.
    
    """
    pass




def newton_e():
    """Newton's method using exact derivative provided by user.
    
    
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
    """
    pass




def horner_method(f, x):
    """
    Horner's Method or synethic division.
    
    Applies synethic division to polynomial equation and returns coefficients
    plus the remainder. 
    
    This is also known as deflation of the function.

    Parameters
    ----------
    f : list
        polynomial coefficients in order of increasing exponent power
        a_1*x^4 + a_2*x^3 + a_3*x^2 + a_4*x + a_5
        f = [a_5, a_4, a_3, a_2, a_1]
    x : float
        initial estimate of root 

    Returns
    -------
    g : list
        polynomical coefficients of defleated polynomial in order of increasing
        power
    R : float
        remainder after synethic division

    """

    n = len(f)-1
    g = np.zeros(n+1)
    
    for i in range(n-1, -1, -1):
        g[i] = f[i+1] + x*g[i+1]
        
    R = f[0] + x*g[0]
        
    return g, R






def ndpnm(f, x, abs_error=0.001, max_iter=20):
    """
    N-degree polynomial Newton's method
    
    This method iteratively solves for a root of an n-degree polynomial. 
    It returns the root and the coefficients of the defleated polynomial 
    equation of degree n-1. The defleated polynomial can again be solved 
    while recognizing there is a discontinuity at each root previously 
    determined by this method. 

    Parameters
    ----------
    f : list
        polynomial coefficients in order of increasing exponent power
        a_1*x^4 + a_2*x^3 + a_3*x^2 + a_4*x + a_5
        f = [a_5, a_4, a_3, a_2, a_1]
    x : float
        initial estimate of root 
    abs_error : float
        absolute error, convergence tolerance for root. The default is 0.001.
    max_iter : int
        maximum number of iterations allowed to achieve convergence tolerance.
        The default is 20.

    Returns
    -------
    g : list
        polynomical coefficients of defleated polynomial in order of increasing
        power
    cur_x : float
        a root of the polynomial equation
        
    Usage Notes
    -----------
    #a0, a(0+1), a(0+2), ..., a(n)
    #3rd degree polynomial
    f = np.array([110, -7, -8, 1]) #coefficients in reverse order
    guess_x = 4
    max_iter = 20
    
    g, x = ndpnm(f, guess_x, 0.001, 20)
    x1, x2 = quadratic_roots(g)
    
    print(x, x1, x2)

    """
    
    count = 0
    error = abs_error * 100
    
    while error > abs_error and count < max_iter:
        
        g, R1 = horner_method(f, x)
        h, R2 = horner_method(g, x)
        
        cur_x = x - R1/R2
        
        error = abs(x - cur_x)
        count += 1
        
        x = cur_x
    
    return g, cur_x






def quadratic_roots(f):
    """Compute the roots of a quadratic equation.
    
    Parameters
    ----------
        f : list
            [c, b, a] coefficients of the quadratic equation in reverse order
            
    Returns
    -------
        Tuple(x1, x2)
        
        x1 : float
            first root
        x2 : float
            second root   
    """
    a = f[2]
    b = f[1]
    c = f[0]
    
    x1 = (-b + math.sqrt(b**2-4*a*c))/(2*a)
    x2 = (-b - math.sqrt(b**2-4*a*c))/(2*a)

    return x1, x2