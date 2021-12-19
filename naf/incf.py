"""
Interpolation and Curve Fitting

A collection of fuctions that can be used to create mathematical functions
that model sets of emperical data.

"""

import numpy as np
import math
from naf import linalg

def lagrangian_poly(pts, n, x):
    """Lagrangian polynomial method

    Uses the Lagrangian polynomial method to compute an interpolated valued
    corresponding to the x-value.

    Parameters:
    -----------
    pts : 2D numpy array
        array of (x,y) coordinate pairs of points for the construsted
        polynomial to pass through

    n : integer
        degree of polynomial to be constructed

    x : float
        corresponding value for which to interpolate

    Returns:
    --------
    y : float
        interpolated valued corresponding to x

    Raises:
    -------
    ValueError
    
    Notes:
    ------
    Usage 1: generate list of values

    from functools import partial
    import numpy as np
    from naf.incf import lag_poly

    pts = np.array([[-2.3,2.1],[0.5,-1.3],[3.1,4.2]])
    x = np.linspace(-10,10)

    fp = partial(lag_poly, pts, n)
    f = list(map(fp,x))


    Usage 2: generate single values
    
    from naf.incf import lag_poly
    import numpy as np

    pts = np.array([[-2.3,2.1],[0.5,-1.3],[3.1,4.2]])

    x = -1.0

    f = lag_poly(pts, n, x)

    """

    if n < len(pts[1]) - 1:
        raise ValueError("Not enough points for specified polynomial degree.")

    #check for duplicate points
    #check for divide by zero

    f = 0.0
    npc = n + 1 #number of points considered in interpolate

    for i in range(npc):
        p = 1.0
        for j in range(npc):
            if i != j:
                p = p*(x - pts[j,0])/(pts[i,0] - pts[j,0])
        f = f + p*pts[i,1]

    return f





def neville_poly(pts, x):
    """Neville's Method for polynomial interpolation



    Parameters:
    -----------
    pts : 2D numpy array
        array of (x,y) coordinate pairs to be used in the
        polynomial interpolation
    x : float
        corresponding value for which to interolate    

    Returns:
    --------
    tb : n-D numpy array
        an n x n dimensional array of the original x and y coordinates
        sorted ascending by the difference between x coordinates and the
        x interpolate with the interpolation table appended
        [x-coordinates, y-coordinates, p[i,1], p[i,2], ..., p[i,n]]

    Raises:
    -------

    Notes:
    ------

    Examples:
    ---------
    pts = np.array([[10.1,0.17531],[22.2,0.37784],[32.0,0.52992],[41.6,0.66393],[50.5,0.63608]])

    tb = neville_poly(pts, 27.5)

    print(tb)

    [[0.52992 0.46009 0.462   0.46174 0.45754]
     [0.37784 0.456   0.46072 0.47902 0.     ]
     [0.66393 0.44521 0.55842 0.      0.     ]
     [0.17531 0.37376 0.      0.      0.     ]
     [0.63608 0.      0.      0.      0.     ]]
    """

    n = pts.shape[0]
    diff = abs(x - pts[...,0])
    id_sort = np.argsort(diff,0)
    pts = pts[id_sort]
    xr = pts[...,0]
    yr = pts[...,1]

    tb = np.zeros(shape=(n,n))
    tb[...,0] = yr

    #moves down columns and then across rows
    #column j and then down the rows i 
    for j in range(1,n):
        for i in range(n):
            if i+j >= n:
                tb[i,j] = 0.0
            else:
                tb[i,j] = ((x - xr[i])*tb[i+1,j-1] + (xr[i+j] - x)*tb[i,j-1])/(xr[i+j] - xr[i])
            
    tb = np.column_stack((xr, tb))
    
    return tb 


############################################
# Divided Differences Polynomial Evaluation
############################################



def dd_tb(pts):
    """Constructs a divided difference table from a set of points.

    Computes the divided difference table for a set of unevenly spaced
    data points. This function can also be used for evenly spacing
    data points

    The od_tb function will be slightly faster due to 
    some savings in computation. 

    Parameters:
    ----------
    pts : 2D numpy array
        array of (x,y) coordinate pairs to be used in the
        divided difference table

    Returns:
    --------
    tb : n-D numpy array
        an n x n dimensional array of the divided differences
        starting with the zero-order difference

    Raises:
    -------

    Notes:
    ------

    Examples:
    ---------
   """ 
    n = pts.shape[0]
    xr = pts[...,0]
    yr = pts[...,1]
    
    tb = np.zeros((n,n))
    tb[...,0] = yr
    
    for j in range(1,n):
        for i in range(0,n-j):
            tb[i,j] = (tb[i+1,j-1] - tb[i,j-1])/(xr[j+i] - xr[i])
            
    return tb







def dd_poly(f, xr, x, n):
    """Polynomial evaluation from divided difference table
    
    Uses nested multiplication to evaluate the polynomial approximated
    by the divided difference method. 
    
    Parameters:
    -----------
    f : 1D numpy array
        array of divided differences from divided difference table
        (often the top row of the divided difference table, unless the
         user is trying to center the x-value in the table of value)
    xr : 1D numpy array
        corresponding array of x-values from divided difference table
    x : float
        value for which to interpolate
    n : interger
        degree of polynomial for interpolation
        
    Returns:
    --------
    y : float
        interpolated value
        
    Raises:
    -------
    
    Notes:
    ------
    
    1.  If the x-values are sorted ascending in the divided difference
        table the user can center the value for which they are interpolating.
        This can be done by determining the subset of rows where the x-value
        is centered, either manually or programtically, and then setting
        the 'f' variable to first row of that subset of data.
        (See Ex. 2 below)

    Examples:
    ---------
    Ex. 1:

    Interpolating the value of the function f(x) = exp(x) at x=0.2 given 
    three points below. 
    
    pts = np.array([[0,1],[0.1,1.1052],[0.3,1.3499]])
    tb = dd_tb(pts)
    
    x = 0.2
    f = tb[0]
    xr = pts[...,0]
    n = 2
    
    y_interpolate = dd_eval(f, xr, x, n)
    
    print(y_interpolate)
    
    >>>1.2218333333333333

    Ex. 2
    
    """
    
    y = 0.0
    
    for i in range(n,0,-1):
        y = (y + f[i])*(x-xr[i-1])
        
    y = y + f[0]
    
    return y


#############################################
# Ordinary Differences Polynomial Evaluation
#############################################




def od_tb(f):
    """Constructs an ordinary difference table from a set of points.

    Parameters:
    ----------
    f : 1D numpy array
        array of function values corresponding to evenly spaced
        x-values

    Returns:
    --------
    tb : n-D numpy array
        an n x n dimensional array of the differences
        starting with the zero-order difference

    Raises:
    -------

    Notes:
    ------

    Examples:
    ---------
   """ 
    n = f.shape[0]
    
    tb = np.zeros((n,n))
    tb[...,0] = f
    
    for j in range(1,n):
        for i in range(0,n-j):
            tb[i,j] = tb[i+1,j-1] - tb[i,j-1]
            
    return tb






def od_poly(df, x0, h, x, n):
    """Polynomial evaluation from ordinary difference table.

    Uses nested multiplication to evaluate the polynomial approximated
    by the ordinary difference method. 

    Parameters:
    -----------
    df : 1D numpy array
        array of ordinary differences from ordinary difference table
        (often the top row of the ordinary difference table, unless the
         user is trying to center the x-value in the table of value)
    x0 : float
        first x-value corresponding to array for ordinary differences
        selected from the od table
    h : float
        delta-x, the constant spacing of the x-value
    x : float
        value for which to interpolate
    n : integer
        degree of polynomial
        

    Returns:
    --------
    y : float
        interpolated value

    Raises:
    -------

    Notes:
    ------

    1.  If the x-values are sorted ascending in the divided difference
        table the user can center the value for which they are interpolating.
        This can be done by determining the subset of rows where the x-value
        is centered, either manually or programtically, and then setting
        the 'f' variable to first row of that subset of data.
        (See Ex. 1 below)

    Examples:
    ---------

    Ex. 1:

    import numpy as np
    from naf.incf import od_tb, od_poly
    from tabulate import tabulate 

    pts = np.array([[0.0,0.0],[0.2,0.203],[0.4,0.423],
                    [0.6,0.684],[0.8,1.030],[1.0,1.557],[1.2,2.572]])
    yr = pts[...,1]
    
    tbr = od_tb(yr)
    
    tb = np.column_stack((pts[...,0],tbr))
    
    tb_h = ['x', 'f(x)', 'Df', 'D2f', 'D3f', 'D4f', 'D5f', 'D6f']
    print(tabulate(tb, tb_h))  

    >>>
          x    f(x)     Df    D2f    D3f    D4f    D5f    D6f
        ---  ------  -----  -----  -----  -----  -----  -----
        0     0      0.203  0.017  0.024  0.02   0.032  0.127
        0.2   0.203  0.22   0.041  0.044  0.052  0.159  0
        0.4   0.423  0.261  0.085  0.096  0.211  0      0
        0.6   0.684  0.346  0.181  0.307  0      0      0
        0.8   1.03   0.527  0.488  0      0      0      0
        1     1.557  1.015  0      0      0      0      0
        1.2   2.572  0      0      0      0      0      0

    Given the table above find the f(0.73) from a cubic interpolating polynomial. 

    The example in the book does not extrapolate using the first four values of the 
    table to get a cubic. Instead it centers x=0.73 in the data and then interpolates.

    df = tbr[2]
    x0 = 0.4
    h = 0.2
    x = 0.73
    n = 3
    
    y_intp = od_poly(df, x0, h, x, n)
    
    print(y_intp)

    >>>0.89322525
    """
    
    y = 0.0
    s = (x - x0)/h
    
    for i in range(n,0,-1):
        y = (y + df[i]/math.factorial(i))*(s-i+1)
        
    y = y + df[0]
    
    return y


def bnc(n,j):
    """Binomial coefficient by the product rule

    Parameters:
    -----------
    n : integer

    j : integer

    Returns:
    --------
    bnc : integer
        binomial coefficient
    """
    bnc = 1.0
    
    for j in range(0,n+1):
        bnc = bnc * (n+1-(j+1))/(j+1)

    return bnc


def od_value(f, n, i):
    """Compute a specific ordinary difference value

    Computes a specific ordinary difference value without
    computing the entire table of values

    Parameters:
    -----------
    f : 1D numpy array
        array of function values corresponding to evenly spaced
        x-values
    n : integer
        degree of ordinary difference value to be calculated
    i : integer
        row index of ordinary difference value to to be calculated

    Returns:
    --------
    df : float
        ordinary difference value of degree n at row index i

    """
    #there might be a cleaner way to do this with nested 
    #multiplication but this method appears to work
    
    #first term of binomial coefficient
    bnc = 1.0 

    for j in range(0,n+1):
        if j == 0:
            df = bnc*f[i+n]
        elif j == 1:
            df = df - bnc*f[i+n-j]
        else:
            if j%2 == 0:
                df = df + bnc*f[i+n-j]
            if j%2 == 1:
                df = df - bnc*f[i+n-j]
        #binomical coefficient by product rule
        #product i=1 to n: (n+1-i)/i
        bnc = bnc * (n+1-(j+1))/(j+1)

    return df




def newton_gregory_poly(xi,x,h,n,i,f):
    """Newton-Gregory forward difference method

    Parameters:
    -----------
    xi : float
        first value x-value used in table
    x : float
        value to for which to interpolate
    h : float
        equal spacing of x values (x1-x0)
    n : integer
        degree of interpolating polynomial
    i : integer
        starting index in array of data
    f : 1D numpy array
        array of function values corresponding to evenly spaced
        x-values

    Returns:
    --------
    p : float
        interpolated value
    """

    s = (x-xi)/h
    p = 0.0

    for j in range(n,0,-1):
        p = (p + od_value(f,j,i)/math.factorial(j))*(s-(j-1))

    p = p + od_value(f,0,i)

    return p



#################################
# Cubic Spline Interpolation
#################################



def c_h(x):
    """Compute array of h-values.
    
    Parameters
    ----------
    x : 1D numpy array, float
        array of x-values of defined points
        
    Returns
    -------
    h : 1D numpy array, float
        the differences between successive defined points
    """
    n = x.size - 1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i+1] - x[i]
    
    return h

def cubic_spline_coeff_matrix(x, h, end_condition):
    """Assemble the tri-diagonal coefficient matrix for cubic splines.

    Parameters:
    -----------
    x : 1D numpy array, float
        array of x-values of defined points
    end_condition : integer
        integer from 1 to 4 indicating the end condition to be used
        1 - cubic spline approach linearity at ends, S0 = 0 and Sn = 0
        2 - end slopes forced to specific values A and B
        3 - cubic spline approach parbolas at ends, S0=S1 and Sn = Sn-1
        4 - extrapolates S0 from S1 and S2, and Sn-2 from Sn-1 and Sn;
            spline match f(x) exactly if f(x) is a cubic

    Returns:
    --------
    csm : 2D numpy array, float
        tri-daigonal matrix of coefficients for cubic spline

    """

    n = x.size - 1

    if end_condition == 1:
        c1 = 2*(h[0] + h[1])
        c2 = h[1]
        c3 = h[n-2]
        c4 = 2*(h[n-2] + h[n-1])
    if end_condition == 2:
        c1 = 2*h[0]
        c2 = h[0]
        c3 = h[n-1]
        c4 = 2*h[n-1]
    if end_condition == 3:
        c1 = 3*h[0] + 2*h[1]
        c2 = h[1]
        c3 = h[n-2]
        c4 = 2*h[n-2] + 3*h[n-1]
    if end_condition == 4:
        c1 = ((h[0] + h[1])*(h[0] + 2*h[1]))/h[1]
        c2 = (h[1]**2 - h[0]**2)/h[1]
        c3 = (h[n-2]**2 - h[n-1]**2)/h[n-2]
        c4 = ((h[n-1] + h[n-2])*(h[n-1] + 2*h[n-2]))/h[n-2]

    if end_condition in (1,3,4):
        csm = np.zeros((n-1,3))
        for i in range(n-1):
            if i == 0:
                csm[i][0] = 0.0
                csm[i][1] = c1
                csm[i][2] = c2
            elif i == n-2:
                csm[i][0] = c3
                csm[i][1] = c4
                csm[i][2] = 0.0
            else:
                csm[i][0] = h[i]
                csm[i][1] = 2*(h[i] + h[i+1])
                csm[i][2] = h[i+1]
    elif end_condition == 2:
        csm = np.zeros((n+1,3))
        
        csm[0][0] = 0.0
        csm[0][1] = c1
        csm[0][2] = c2
        
        for i in range(0,n-1):
            csm[i+1][0] = h[i]
            csm[i+1][1] = 2*(h[i] + h[i+1])
            csm[i+1][2] = h[i+1]
                
        csm[n][0] = c3
        csm[n][1] = c4
        csm[n][2] = 0.0

    return csm

def cubic_spline_vector(pts, h, end_condition, A=0, B=0):
    """Assemble cubic spline right-hand side vector.
    
    Parameters:
    -----------
    pts : 2D numpy array, float
        array of xy coordinate pairs to be fitted
    end_condition : integer
        integer from 1 to 4 indicating the end condition to be used
        1 - cubic spline approach linearity at ends, S0 = 0 and Sn = 0
        2 - end slopes forced to specific values A and B
        3 - cubic spline approach parbolas at ends, S0=S1 and Sn = Sn-1
        4 - extrapolates S0 from S1 and S2, and Sn-2 from Sn-1 and Sn;
            spline match f(x) exactly if f(x) is a cubic
    A (optional) : float
        slope at beginning of spline
    B (optional) : float
        slope at end of spline
        
    Returns:
    --------
    b : 1D numpy array, float
        array of values for right-hand side
        
    """
    x = pts[...,0]
    y = pts[...,1]
    
    n = x.size - 1
    
    if end_condition != 2:
        b = np.zeros(n-1)

        #watch the indexing of b, need to offset to the prior index
        for i in range(1,n):
            b[i-1] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
            
    if end_condition == 2: 
        b = np.zeros(n+1)
        
        b[0] = 6*((y[1]-y[0])/h[0] - A)

        for i in range(1,n):
            b[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
            
        b[n] = 6*((y[n]-y[n-1])/h[n-1] - B)
        
    return b

def solve_s_vector(csm, h, b, end_condition):
    """Solves for the S-vector and adds correct end conditions.
    
    Parameters
    ----------
    csm : 2D numpy array, float
        tridiagonal matrix of cubic spline coefficients
    b : 1D numpy array, float
        right-hand side vector
    end_condition : integer
        integer from 1 to 4 indicating the end condition to be used
        1 - cubic spline approach linearity at ends, S0 = 0 and Sn = 0
        2 - end slopes forced to specific values A and B
        3 - cubic spline approach parbolas at ends, S0=S1 and Sn = Sn-1
        4 - extrapolates S0 from S1 and S2, and Sn-2 from Sn-1 and Sn;
            spline match f(x) exactly if f(x) is a cubic
            
    Returns
    -------
    s : 1D numpy array, float
        S-value solution array
        
    """
    s = linalg.tdqsv(csm, b)
    
    if end_condition == 1:
        s = np.insert(s, 0, 0)
        s = np.append(s, 0)
    if end_condition == 2:
        pass
    if end_condition == 3:
        s = np.insert(s, 0, s[0])
        s = np.append(s, s[-1])
    if end_condition == 4:
        #s[0] = S_1 and s[1] = S_2 due to indexing
        s0 = ((h[0] + h[1])*s[0] - h[0]*s[1])/h[1]
        #s[-1] = S_n-1 and s[-2] = S_n-2
        sn = ((h[-2] + h[-1])*s[-1] - h[-1]*s[-2])/h[-2]
        s = np.insert(s, 0, s0)
        s = np.append(s, sn)
        
    return s

def cubic_spline_poly_coeffs(s, y, h):
    """Calculates the polynomial coefficients for each internal.
    
    Uses the S-vector to calculate the polynomial coefficients for each
    interval.
    
    Parameters
    ----------
    s : 1D numpy array, float
        S-value solution array
    y : 1D numpy array, float
        array of y-values of defined points
        
    Returns
    -------
    a, b, c, d : 2D numpy array, float
        array of polynomial coefficients for each interval
        
    """   
    k = s.size - 1
    
    a = np.zeros(k)
    b = np.zeros(k)
    c = np.zeros(k)
    d = y[0:-1]
    
    for i in range(0,k):
        a[i] = (s[i+1] - s[i])/(6*h[i])
        b[i] = s[i]/2
        c[i] = (y[i+1] - y[i])/h[i] - (2*h[i]*s[i] + h[i]*s[i+1])/6
        
    return np.array([a,b,c,d])

def cubic_spline_interpolation(csc, ix, x):
    """Performs an interpolation for a given x-value along the defined spline.
    
    Determines if the x-interpolate is in the defined range. Then, determines 
    the interval where the x-interpolate is defined. Given the interval, the
    polynomial coefficients for that interval are selected and used to 
    interpolate the function value, iy, for the x-interpolate.
    
    Parameters
    ----------
    csc : 2D numpy array, float
        array of polynomial coefficients for each interval
    ix : float
        x-interpolate value
    x : 1D numpy array, float
        array of x-values of defined points
    
    Returns
    -------
    iy : float
        y-interpolate value
    """
    n = x.size - 1
    
    #check that ix is in the data range
    ind = None
    if ix >= x[0] and ix <= x[n]:
        #find the interval for ix
        for i in range(1,n+1):
            if x[i] >= ix:
                ind = i - 1
                break
    else:
        raise ValueError("X-interpolation value outside range.")
                
    a = csc[0][ind]
    b = csc[1][ind]
    c = csc[2][ind]
    d = csc[3][ind]
    
    #more computationally efficient way to write ax^3 + bx^2 + cx + d
    x = ix - x[ind]
    iy = ((x*a + b)*x + c)*x + d
    
    return iy
    

def csisv(ixv, pts, end_condition, A=0, B=0):
    """Calculates the cubic spline and the y-interpolates.
    
    This function calculates the cubic spline polynomial coefficients
    for each interval. Then, it calculates and returns an array of
    y-interpolates for the given x-interpolates.
    
    Hint: It can be used for a single value by providing an array 
    with one value. Note, it will return an array with one value.
    
    Paramaters
    ----------
    ixv : 1D numpy array, float
        array of x-interpolate values
    pts : 2D numpy array, float
        array of defined xy points
    end_condition : integer
        integer from 1 to 4 indicating the end condition to be used
        1 - cubic spline approach linearity at ends, S0 = 0 and Sn = 0
        2 - end slopes forced to specific values A and B
        3 - cubic spline approach parbolas at ends, S0=S1 and Sn = Sn-1
        4 - extrapolates S0 from S1 and S2, and Sn-2 from Sn-1 and Sn;
            spline match f(x) exactly if f(x) is a cubic
    
    Returns
    -------
    iyv : 1D nupy array, float
        array of y-interpolate values
    """
    x = pts[...,0]
    y = pts[...,1]
    
    h = c_h(x)
        
    csm = cubic_spline_coeff_matrix(x, h, end_condition)
    b = cubic_spline_vector(pts, h, end_condition, A, B)
    s = solve_s_vector(csm, h, b, end_condition)
    csc = cubic_spline_poly_coeffs(s, y, h)
    
    iyv = np.empty(ixv.size)
    for i in range(ixv.size):
        iyv[i] = cubic_spline_interpolation(csc, ixv[i], x)
    
    return iyv
