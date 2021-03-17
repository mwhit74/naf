
import numpy as np
import pdb

"""Naming Convention

First two letters are the matrix type

ge - generic
td - tridiagonal

Second two letters are the method used

ga - gaussian elimination
fe - forward elimination
bs - back substitution

"""

def gega(a):
    """Generic square matrix LU Decomposition using Gaussian elmination with 
    partial pivoting.

    This function factors the a matrix into two matricies L and U. L is a lower
    triangular matrix of the multiplication coefficients with 1's on the
    diagonal. U is an upper triangular matrix of the remaining coefficients of
    the original a matrix after pivoting and row reduction.

    This function does the decomposition in place. That is to say it stores the
    resulting values in the a matrix. 

    This function also keeps track of any row exchanges due to partial pivoting
    of the rows. 

    Paremeters
    ----------
    a : 2D numpy array 
        an n x n matrix to be factored into LU

    Raises
    ------
    Exception
        If the matrix is not square it is not invertable.
    
    Returns
    -------
    a : 2D numpy array
        combined L and U matricies; L is the lower triangular with 1's on 
        the diagonal; U is the upper triangular including the values on 
        the diagonal
    ov : 1D numpy array 
        row order vector; keeps track of the reordering due to 
        partial pivoting of the rows

    """
    n,m = a.shape
    ov = np.arange(n) #order_vector
    
    if n != m:
        raise Exception("Matrix is not sqaure.")
    
    #row reduction
    for j in range(n):
        pvt = abs(a[ov[j],j]) #gets current pvt on diagonal
        new_pvt_row = None
        org_pvt_row = j #keeps track of row location of pvt
    
        #cycle thru entries in first column
        #find largest value
        #aka finding the max pivot
        for i in range(j+1,n):
            if abs(a[ov[i],j]) > pvt:
                pvt = abs(a[ov[i],j])
                new_pvt_row = i
    
        #switch largest value to be pivot
        #this reduces rounding error
        if (new_pvt_row != None and org_pvt_row != new_pvt_row):
            ov[org_pvt_row] = new_pvt_row
            ov[new_pvt_row] = org_pvt_row
    
        #calculates multipliers for row reduction
        for i in range(j+1,n):
            a[ov[i],j] = a[ov[i],j]/a[ov[j],j]

    
        #creates zeros below the main diagonal
        for i in range(j+1,n): #row number
            for k in range(j+1,n): #column number
                a[ov[i],k] = a[ov[i],k] - a[ov[i],j]*a[ov[j],k]
    
    return a, ov #where a is now L in the lower and U in the upper




def tdga(a):
    """Tridiagonal matrix LU decomposition using Gaussian elimination without
    partial pivoting.

    This function factors the a matrix into two matricies L and U. L is a lower
    triangular matrix of the multiplication coefficients with 1's on the
    diagonal. U is an upper triangular matrix of the remaining coefficients of
    the original a matrix after pivoting and row reduction.

    This function does the decomposition in place. That is to say it stores the
    resulting values in the a matrix. 

    Parameters
    ----------
    a : 2D numpy array
        n x 2 matrix which is a compressed form of a tridiagonal matrix

    Returns
    -------
    a : 2D numpy array
        n x 2 matrix which is a compressed form of a tridiagonal matrix;
        combined L and U matricies; L is the lower triangular with 1's on 
        the diagonal; U is the upper triangular including the values on 
        the diagonal

    """
    
    n, m = a.shape
    
    if m != 4:
        raise Exception('a tridiagonal matrix is represented as an n x 3.')
    
    for i in range(1,n):
        a[i,1] = a[i,1]/a[i-1,2]
        a[i,2] = a[i,2] - a[i,1]*a[i-1,3]    
    
    return a





def gefe(lu, ov, b):
    """Generic matrix forward elmination applied to the b vector.

    Applies the operations recorded in the decomposition, the L lower triangular
    matrix and the row exchanges, to the b vector. 

    This step can be taken in the lu decomposition function but there are
    significant efficienies to be gained by having a separate function if there
    are multiple b vectors to be considered which is very often the case. 

    Parameters
    ----------
    lu : 2D numpy array
         factored A matrix; L is the lower triangular with 1's on the 
         diagonal; U is the upper triangular including the values on the 
         diagonal
    b : 1D numpy array
        the right hand side; the constant values in the equations
    ov : 1D numpy array
         row order vector; keeps track of the reordering due to 
        partial pivoting of the rows

    Returns
    -------
    b : 1D numpy array
        intermediate solution vector
        the b vector with the factored operations from the decomposition 
        applied; often referred to as 'c' in literature on the subject

    """
    n,m = lu.shape
    for j in range(n):
        for i in range(j+1,n):
            b[ov[i]] = b[ov[i]] - lu[ov[i],j]*b[ov[j]]

    return b



def tdfe(LU, b):
    """
    Tridiagonal matrix forward substitution.

    Parameters
    ----------
    lu : 2D numpy array
         factored A matrix; L is the lower triangular with 1's on the 
         diagonal; U is the upper triangular including the values on the 
         diagonal
    b : 1D numpy array
        the right hand side; the constant values in the equations

    Returns
    -------
    None.

    """
    pass


def gebs(lu, ov, b):
    """
    Generic matrix back substituation applied to solve for the x vector.

    Solves the upper triangular matrix U with the factored b vector for the x
    vector. 

    Paramters
    ---------
    lu : 2D numpy array
        factored A matrix; L is the lower triangular with 1's on the 
        diagonal; U is the upper triangular including the values on the diagonal
    b : 1D numpy array
        intermediate solution vector;
        the b vector with the factored operations from the decomposition 
        applied; often referred to as 'c' in literature on the subject
    ov : 1D numpy array
        row order vector; keeps track of the reordering due to 
        partial pivoting of the rows

    Retuns
    ------
    x : 1D numpy array
        the solution vector
    
    """
    m, n = lu.shape
    mmo = m - 1
    x = np.empty((n,1),dtype='float')   
    #back substitution
    x[ov[mmo]] = b[ov[mmo]]/lu[ov[mmo],mmo] # last row solution
    for j in range(mmo,-1,-1):
        x[ov[j]] = b[ov[j]] #initialize solution value
        for k in range(mmo,j,-1):
            #group known terms in numerator
            x[ov[j]] = x[ov[j]] - x[ov[k]]*lu[ov[j],k] 
        x[ov[j]] = x[ov[j]]/lu[ov[j],j] #solving equation by division
    
    return x


def gesv(lu, ov, b):
    """Generic matrix application of the forward elimination and back 
    substituation steps.
    

    Parameters
    ----------
    lu : 2D numpy array
         matrix decomposed into upper and lower operation coefficients for
         forward and back substitution
    ov : 1D numpy arry
         row order vector; keeps track of the reordering due to 
         partial pivoting of the rows
    b : 1D numpy array
        the right hand side; the constant values in the equations

    Returns
    -------
    x : 1D numpy array
        solution vector

    """

    b = gefe(lu, ov, b)
    x = gebs(lu, ov, b)
    return x 






def ddm(a):
    """
    Permutate matrix to be diagonally dominate if possible. 
    
    Checks for a square matrix.
    
    Checks for strict diagonal dominance per the following formula
    
    abs(a[i,i]) > sum(abs(a[i,j])) for j = 0 to n and j != i, i = 0, 1, ... ,n

    Parameters
    ----------
    a : 2D numpy array
        maxtrix to make diagonally dominate

    Raises
    ------
    Exception 1
        Raises Exception if matrix is not square.
    Exception 2
        Raises Exception if matrix cannot be permutated to a diagonally
        dominate matrix

    Returns
    -------
    a : 2D numpy array
        permutation of the original matrix that is diagonally dominate
    ov : 1D numpy array
        order vector of diagonally dominate matrix permutation

    """
    m, n = a.shape
    
    if n != m:
        raise Exception('Matrix is not square.')
    
    p = np.absolute(a)
    
    q = np.amax(p, 1)
    r = np.argmax(p, 1)
    s = np.sum(p, 1)
    t = np.subtract(s, q)
    
    u = np.sort(r)
    n,m = p.shape
    v = np.arange(n)
    
    #check if matrix is strictly diagonally dominate
    if np.all(q > t) and np.all(u == v):
        #perumtate matrix to diagonally dominate
        ov = np.arange(n)
        for i in range(n):
            ov[r[i]] = i
        
        return a[ov,:], ov
    
    else:
        raise Exception('Matrix cannot be strictly diagonally dominate')
    





def geji(a, x1, b, tol = 0.0001, max_iter = 50):
    """
    Jacobian method: iterative solution to set of linear equations
    
    Uses fixed-point iteration on a set of equations to iteratively find
    the solution.
    
    Checks for a square matrix. Raises an exception for a non-square matrix
    
    Checks for strict diagonal dominance to ensure convergence. Raises an 
    exception if the matrix cannot be permutated to a strict diagonally
    dominate matrix.

    Parameters
    ----------
    a : 2D numpy array
        an n x n coefficient matrix of the linear equations to be solved
    x1 : 1D numpy array
        initial estimate of solution vector
    b : 1D numpy array
        the right hand side; the constant values in the equations
    tol : float, optional
        the convergence tolerance of the solution vector. The default is 0.0001.
    max_iter : integer, optional
        maximum number of iterations to perform if the solution is not
        converging. The default is 50.
        
    Raises
    ------
    Exception1 - non-square matrix
        Forwards exception from ddm function
    Exception2 - non-diagonally dominate matrix
        Forwards exception from ddm function

    Returns
    -------
    x2 : 1D numpy array
        solution vector with an error tolerance of tol. Vector is reordered
        to match original order of equations before permutations to create
        a diagonally dominate matrix
    verror : 1D numpy array
        error vector; error of each solution. Vector is reordered
        to match original order of equations before permutations to create
        a diagonally dominate matrix
    num_iter : integer
        number of iterations required to converge.

    """
    m, n = a.shape
    
    try:
        a, ov = ddm(a)
    except Exception as e:
        raise e
        
    x1 = x1[ov]
    b = b[ov]
    
    x2 = np.zeros(n)
    vtol = np.repeat(tol, m)
    verror = np.repeat(tol+1, m)
    
    for i in range(m):
        b[i] = b[i]/a[i,i]
        x2[i] = x1[i]
        for j in range(m):
            if j != i:
                a[i,j] = a[i,j]/a[i,i]
    
    num_iter = 0
                
    while np.all(verror > vtol) and num_iter < max_iter:
        for i in range(m):
            x1[i] = x2[i]
            x2[i] = b[i]

        for i in range(m):
            for j in range(m):
                if j != i:
                    x2[i] = x2[i] - a[i,j]*x1[j]
                    
        verror = np.absolute(np.subtract(x2, x1))
        
        num_iter += 1
    
    #return to original equation order
    xt = np.zeros(ov.size)
    vet = np.zeros(ov.size)
    for i in range(ov.size):
        xt[i] = x2[ov[i]]
        vet[i] = verror[ov[i]]
        
    
    return xt, vet, num_iter







def gegs(a, x1, b, w=1.0, tol = 0.0001, max_iter = 50):
    """
    Gauss-Seidel method: iterative solution to set of linear equations with 
    optional over-relaxation 

    Parameters
    ----------
    a : 2D numpy array
        an n x n coefficient matrix of the linear equations to be solved
    x1 : 1D numpy array
        initial estimate of solution vector
    b : 1D numpy array
        the right hand side; the constant values in the equations
    w : float
        over-relaxation factor; 1.0 <= w <= 2.0. The default is 1.0, no 
        over-relaxation. 
    tol : float, optional
        the convergence tolerance of the solution vector. The default is 0.0001.
    max_iter : integer, optional
        maximum number of iterations to perform if the solution is not
        converging. The default is 50.

    Raises
    ------
    Exception1 - non-square matrix
        Forwards exception from ddm function
    Exception2 - non-diagonally dominate matrix
        Forwards exception from ddm function

    Returns
    -------
    x1 : 1D numpy array
        solution vector with an error tolerance of tol. Vector is reordered
        to match original order of equations before permutations to create
        a diagonally dominate matrix
    verror : 1D numpy array
        error vector; error of each solution. Vector is reordered
        to match original order of equations before permutations to create
        a diagonally dominate matrix
    num_iter : integer
        number of iterations required to converge.
    """
    
    m, n = a.shape
    
    try:
        a, ov = ddm(a)
    except Exception as e:
        raise e
        
    x1 = x1[ov]
    b = b[ov]
    
    vtol = np.repeat(tol, m)
    verror = np.repeat(tol+1,m)
    
    for i in range(m):
        b[i] = b[i]*w/a[i,i]
        for j in range(m):
            if j != i:
                a[i,j] = a[i,j]*w/a[i,i]
                
    for i in range(m):
        a[i,i] = w*a[i,i]/a[i,i]
    
    num_iter = 0
    x0 = np.zeros(ov.size)
    
    while np.all(verror > vtol) and num_iter < max_iter:
        x0 = np.copy(x1)
        for i in range(n):
            x1[i] = x1[i] + b[i]
            for j in range(n):
                    if i == j:
                        x1[i] = x1[i] - a[i,j]*x0[j]
                    else:
                        x1[i] = x1[i] - a[i,j]*x1[j]
        
        verror = np.absolute(np.subtract(x1, x0))
        
        num_iter += 1
                    
        
    xt = np.zeros(ov.size)
    vet = np.zeros(ov.size)
    for i in range(ov.size):
        xt[i] = x1[ov[i]]
        vet[i] = verror[ov[i]]
                    
    return x1, verror, num_iter

