"""
Interpolation and Curve Fitting

A collection of fuctions that can be used to create mathematical functions
that model sets of emperical data.

"""

import numpy as np

def lagrangian_poly(pts, n, x):
    """Lagrangian polynomial method

    Uses the Lagrangian polynomial method to create a polynomial that 
    passes to the given coordinate points.

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

