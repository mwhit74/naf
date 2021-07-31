"""
Interpolation and Curve Fitting

A collection of fuctions that can be used to create mathematical functions
that model sets of emperical data.

"""

import numpy as np

def lag_poly(pts, x):
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

    Returns:
    --------
    pc : 1D numpy array
        array of polynomial coefficients in ascending order 
        [a0, a1, a2, ... , an]

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

    fp = partial(lag_poly, pts)
    f = list(map(fp,r))


    Usage 2: generate single values
    
    from naf.incf import lag_poly
    import numpy as np

    pts = np.array([[-2.3,2.1],[0.5,-1.3],[3.1,4.2]])

    x = -1.0

    f = lag_poly(pts, x)

    """

    if n > len(pts[1]) - 1:
        raise ValueError("Not enough points for specified polynomial degree.")

    #check for duplicate points
    #check for divide by zero

    f = 0.0
    num_pts = len(pts[1]) + 1

    for i in range(num_pts):
        p = 1.0
        for j in range(num_pts):
            if i != j:
                p = p*(x - pts[j,0])/(pts[i,0] - pts[j,0])
        f = f + p*pts[i,1]

    return f


