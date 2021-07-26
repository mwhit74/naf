
import numpy as np
import pdb
from warnings import warn

"""Naming Convention

First two letters are the matrix type

ge - generic
td - tridiagonal

Second two letters are the method used

do - doolittle reduction
cr - crout reduction
fe - forward elimination
bs - back substitution

"""




#######################
# DOOLITTLE REDUCTION
#######################



def gedo(a, pivot=True):
    """Generic square matrix LU Decomposition using Doolittle algorithm with
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
    pivot : boolean, optional
        a flag to turn partial pivoting on or off. The default is True. This
        means partial pivoting is ON by default.

    Raises
    ------
    Exception
        Non-invertible matrix. Could be not square or singular. 
    
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
    a = a.copy()
    n,m = a.shape
    ov = np.arange(n) #order_vector
    
    if n != m:
        raise Exception("Matrix is not sqaure.")
    
    #row reduction
    for j in range(0,n):
        if pivot:
            pvt = abs(a[ov[j],j]) #gets current pvt on diagonal
            new_pvt_row_index = None
            org_pvt_row_index = j #keeps track of org pivot row index location
        
            #cycle thru entries in first column
            #find largest value
            #aka finding the max pivot
            for i in range(j+1,n):
                if abs(a[ov[i],j]) > pvt:
                    pvt = abs(a[ov[i],j])
                    new_pvt_row_index = i #keeps track of new pivot row index location
        
            #switch largest value to be pivot
            #this reduces rounding error
            if (new_pvt_row_index != None and org_pvt_row_index != new_pvt_row_index):
                org_pvt_row_num = ov[org_pvt_row_index] #resolves index to actual row number
                new_pvt_row_num = ov[new_pvt_row_index]
                ov[org_pvt_row_index] = new_pvt_row_num #assigns actual row number to new index location
                ov[new_pvt_row_index] = org_pvt_row_num
            
        #checking for no solutions, more unknowns than equations,
        #linear dependence, a singular matrix
        if np.isclose(a[ov[j], j],0.0, atol=1e-03):
            msg = ('Approximate zero value on diagonal; matrix is at least '
                   'unstable and could be singular or could have '
                   'an infinite number of solutions')
            warn(msg)
    
        #calculates multipliers for row reduction
        for i in range(j+1,n):
            try:
                a[ov[i],j] = a[ov[i],j]/a[ov[j],j]
            except ZeroDivisionError as e:
                print("Division by zero error: zero encounter on diagonal, matrix is singular")
                sys.exit()

    
        #creates zeros below the main diagonal
        for i in range(j+1,n): #row number
            for k in range(j+1,n): #column number

                a[ov[i],k] = a[ov[i],k] - a[ov[i],j]*a[ov[j],k]
    
    return a, ov #where a is now L in the lower and U in the upper


def dofe(lu, ov, b):
    """
    Doolittle reduction forward elmination applied to the b vector.

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
    c : 1D numpy array
        intermediate solution vector
        the b vector with the factored operations from the decomposition 
        applied; often referred to as 'c' in literature on the subject

    """
    c = b.copy()
    n,m = lu.shape
    for j in range(n):
        for i in range(j+1,n):
            c[ov[i]] = c[ov[i]] - lu[ov[i],j]*c[ov[j]]

    return c



def dobs(lu, ov, c):
    """
    Doolittle reduction back substituation applied to solve for the x vector.

    Solves the upper triangular matrix U with the factored b vector for the x
    vector. 

    Paramters
    ---------
    lu : 2D numpy array
        factored A matrix; L is the lower triangular with 1's on the 
        diagonal; U is the upper triangular including the values on the diagonal
    c : 1D numpy array
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
    c = c.copy()
    m, n = lu.shape
    mmo = m - 1
    x = np.empty((n),dtype='float')   
    #back substitution
    x[ov[mmo]] = c[ov[mmo]]/lu[ov[mmo],mmo] # last row solution
    for j in range(mmo,-1,-1):
        x[ov[j]] = c[ov[j]] #initialize solution value
        for k in range(mmo,j,-1):
            #group known terms in numerator
            x[ov[j]] = x[ov[j]] - x[ov[k]]*lu[ov[j],k]
        x[ov[j]] = x[ov[j]]/lu[ov[j],j] #solving equation by division
    
    return x


def dosv(lu, ov, b):
    """Doolittle reduction application of the forward elimination and back 
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
    b = b.copy()
    
    c = dofe(lu, ov, b)
    x = dobs(lu, ov, c)
    return x 


def det(lu, ov):
    
    m,n = lu.shape
        
    #count number of swapped rows
    rc = 0
    v = np.arange(ov.size)
    for x,y in zip(ov, v):
        if x != y:
            rc += 1
    #2 swapped rows equals 1 interchange
    rc = rc / 2.0
    
    #multiply entries on diagonal of LU matrix
    det = 1
    for i in range(m):
        det = det*lu[ov[i],i]
        
    #for odd number of row interchanges det * -1
    if rc % 2 != 0:
        det = -1*det
        
    return det







#####################
# FULL INVERSE
#####################




def invr(lu, ov):
    """Uses Doolittle reduction algorithm to compute matrix inverse.

    Solves the LU-decomposition matrix using the Doolittle algorithm. Then,
    successively solves with forward and back substitution each column
    of the idetity matrix. This results in the inverse of the matrix.

    Parameters
    ----------
    lu : 2D numpy array
         matrix decomposed into upper and lower operation coefficients for
         forward and back substitution
    ov : 1D numpy arry
         row order vector; keeps track of the reordering due to 
         partial pivoting of the rows

    Returns
    -------
    inverse : 2D numpy array
        the inverted matrix of matrix 'a'        
    """
    
    n,m = lu.shape
    iden = np.identity(n, dtype=float)
    inverse = np.empty((n,m),dtype=float)

    i = 0
    for col in iden.T:
        x = dosv(lu, ov, col)
        inverse[:,i] = x
        i += 1

    return inverse








######################
# TRI-DIAGONAL 
######################





def tddo(a):
    """Tridiagonal matrix LU decomposition using Doolittle algorithm

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
    a = a.copy()
    n, m = a.shape
    
    if m != 3:
        raise Exception('a tridiagonal matrix is represented as an n x 3.')

    for i in range(1,n):
        a[i,0] = a[i,0]/a[i-1,1]
        a[i,1] = a[i,1] - a[i,0]*a[i-1,2]    
    
    return a

def tdfe(lu, b):
    """
    Doolittle reduction forward elmination applied to the b vector.

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
    c : 1D numpy array
        intermediate solution vector
        the b vector with the factored operations from the decomposition 
        applied; often referred to as 'c' in literature on the subject
    """
    c = b.copy()
    n, m = lu.shape
    
    if m != 3:
        raise Exception('a tridiagonal matrix is represented as an n x 3.')
    
    for i in range(1, n):
        c[i] = c[i] - lu[i,0]*c[i-1]
        
    return c

def tdbs(lu, c):
    """
    Doolittle reduction back substituation applied to solve for the x vector.

    Solves the upper triangular matrix U with the factored b vector for the x
    vector. 

    Paramters
    ---------
    lu : 2D numpy array
        factored A matrix; L is the lower triangular with 1's on the 
        diagonal; U is the upper triangular including the values on the diagonal
    c : 1D numpy array
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
    
    x = c.copy()
    n, m = lu.shape
    
    if m != 3:
        raise Exception('a tridiagonal matrix is represented as an n x 3.')
    
    x[n-1] = x[n-1]/lu[n-1,1]
    
    for i in range(n-2, -1, -1):
        x[i] = (x[i] - lu[i,2]*x[i+1])/lu[i,1]
        
    return x

def tdsv(lu, b):
    """Doolittle reduction application of the forward elimination and back 
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
    b = b.copy()
    
    c = tdfe(lu, b)
    x = tdbs(lu, c)
    return x 



#######################
#   CROUT REDUCTION
#######################



def gecr(a, pivot=True):
    """Generic square matrix LU Decomposition using Doolittle algorithm with
    partial pivoting.
    

    Parameters
    ----------
    a : 2D numpy array
        n x n matrix to be factored into LU
    pivot : boolean, optional
        a flag to turn partial pivoting on or off. The default is True. This
        means partial pivoting is ON by default.

    Raises
    ------
    Exception
        Non-invertible matrix. Could be not square or singular. 
    
    Returns
    -------
    a : 2D numpy array
        combined L and U matricies; L is the lower triangular including the 
        values on the diagonal; U is the upper triangular with 1's on the 
        diagonal
    ov : 1D numpy array 
        row order vector; keeps track of the reordering due to 
        partial pivoting of the rows

    """
    a = a.copy()
    n,m = a.shape
    ov = np.arange(n) #order_vector
    
    if n != m:
        raise Exception("Matrix is not sqaure.")
    
    for j in range(0,n):
        #partial pivot of rows
        if pivot:
            pvt = abs(a[ov[j],j]) #gets current pvt on diagonal
            new_pvt_row_index = None
            org_pvt_row_index = j #keeps track of row location of pvt
        
            #cycle thru entries in first column
            #find largest value
            #aka finding the max pivot
            for i in range(j+1,n):
                if abs(a[ov[i],j]) > pvt:
                    pvt = abs(a[ov[i],j])
                    new_pvt_row_index = i
        
            #switch largest value to be pivot
            #this reduces rounding error
            if (new_pvt_row_index != None and org_pvt_row_index != new_pvt_row_index):
                org_pvt_row_num = ov[org_pvt_row_index]
                new_pvt_row_num = ov[new_pvt_row_index]
                ov[org_pvt_row_index] = new_pvt_row_num
                ov[new_pvt_row_index] = org_pvt_row_num
                
        #checking for no solutions, more unknowns than equations,
        #linear dependence, a singular matrix
        if np.isclose(a[ov[j], j],0.0, atol=1e-03):
            msg = ('Approximate zero value on diagonal; matrix is at least '
                   'unstable and could be singular or could have '
                   'an infinite number of solutions')
            warn(msg)
                
        if j == 0:
            #loop not necessary if in-place storage is used
            #included here for completness
            #for i in range(0,n)
            #   l[i,1] = a[i,1]
            
            for i in range(1,n):
                a[ov[0],i] = a[ov[0],i]/a[ov[0],0]
                
        else:
            for i in range(j,n):
                sum = 0.0
                for k in range(0,j):
                    sum += a[ov[i],k]*a[ov[k],j]
                a[ov[i],j] = a[ov[i],j] - sum
                
            for i in range(j+1, n):
                sum = 0.0
                for k in range(0,j):
                    sum += a[ov[j],k]*a[ov[k],i]
                a[j,i] = (a[ov[j],i] - sum)/a[ov[j],j]
            
    return a, ov


def crfe(lu, ov, b):
    """
    Crout reduction forward elmination applied to the b vector.

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
    
    n, m = lu.shape
    
    c = b.copy()
    
    c[ov[0]] = b[ov[0]] / lu[ov[0],0]
    
    for i in range(1, n):
        for j in range(0, i):
            c[ov[i]] = c[ov[i]] - lu[ov[i],j]*c[ov[j]]
        c[ov[i]] = c[ov[i]]/lu[ov[i],i]
        
    return c


def crbs(lu, ov , c):
    """
    Crout reduction back substituation applied to solve for the x vector.

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
    n,m = lu.shape
    x = np.empty((n,1), dtype=float)
    x = c.copy()
    
    for i in range(n-2, -1, -1):
        for j in range(n-1, i, -1):
            x[ov[i]] = x[ov[i]] - lu[ov[i],j]*x[ov[j]]
            
    return x



def crsv(lu, ov, b):
    """Crout reduction application of the forward elimination and back 
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

    b = b.copy()
    
    c = crfe(lu, ov, b)
    x = crbs(lu, ov, c)
    return x 





#########################
#    ITERATIVE METHODS
#########################


#################################
# Solution for linear equations
#################################




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
        Raises exception from ddm function
    Exception2 - non-diagonally dominate matrix
        Raises exception from ddm function

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
    a = a.copy()
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
    a = a.copy()
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






######################################
# Solutions for nonlinear equations
#####################################



def newton_soe(x0, a_m, b_m, tol= 0.0001, max_iter=20, verbose=False):
    """Netwon's method applied for a system of nonlinear equations
    
    Parameters:
    -----------
    x0 : 1D numpy array
        initial estimate of solution vector
    a_m : function
        function that accepts 1D numpy array of the approx. solution vector
        and returns the Jacobian matrix of the funcitons (a 2D numpy array 
        of the partial derivatives evaluated at the approx. solution vector)
    b_v : function
        function that accepts 1D numpy array of the approx. solution vector
        and returns a 2D numpy array of the functions evaluated at the approx.
        solution vector)
    tol : float, optional
        convergence tolerance. The default is 0.0001
    max_iter : integer, optional
        maximum number of iterations to perform if the solution is not converging.
        The default is 20.
    verbose : bool, optional
        outputs a summary of the solution progression at each solution step
        The default is False
        
    Returns:
    --------
    x0 : 1D numpy array
        approx. solution to the system within the specified tolerance
    dx : 1D numpy array
        solution error w.r.t. the previous approx. solution
    num_iter : integer
        number of iterations required to converge.
        
    Notes:
    ------
    Example a_m function:
    
    def a_m(z):
        x = z[0]
        y = z[1]

        dxf1 = lambda x,y: 2.0*x + 1.0
        dyf1 = lambda x,y: -2.0*y
        dxf2 = lambda x,y: -2.0*x*math.cos(math.pow(x,2))
        dyf2 = lambda x,y: 1.0

        return np.array([[dxf1(x,y),dyf1(x,y)],[dxf2(x,y),dyf2(x,y)]])
    
    Example b_m function:
    
    def b_v(z):
        x = z[0]
        y = z[1]

        f1 = lambda x,y: math.pow(x,2) + x - math.pow(y,2) - 1
        f2 = lambda x,y: y - math.sin(math.pow(x,2))
        return np.array([f1(x,y),f2(x,y)])   
    """

    num_eqs = np.size(x0)
    tol = np.full(num_eqs, 0.0001)
    dx = tol*10
    num_iter = 0
    max_iter = 20

    while np.all(abs(dx)>tol) and num_iter < max_iter:
        a = a_m(x0)

        b = -1*b_v(x0)

        lu, ov = gedo(a)
        dx = dosv(lu, ov, b)[ov]

        x0 = x0 + dx
        num_iter += 1

        if verbose:
            print(f'a: {a}')
            print(f'b: {b}')
            print(f'lu,ov: {lu},{ov}')
            print(f'dx: {dx}')
            print(f'x0: {x0}')
            print(f'num_iter: {num_iter}')
            print('\n')
                
    return x0, dx, num_iter

def newton_soem(x0, a_m, b_m, num_eqs, n, tol= 0.0001, max_iter=20, verbose=False):
    """Modified Newton's method applied to a system of nonlinear equations
    
    The modification in the solution routine is not updating the Jacobian matrix
    of partials with each iteration. Instead the same Jacobian is used to without
    update with the new approximate solution. The Jacobian is updated after n
    iterations with the first Jacobian. Using the same Jacobian for multiple 
    iteration steps saves on re-evaluation the potentially large number of 
    functions in the Jacobian matrix. This is a trade off with the number of 
    steps required to converage.
    
    Parameters:
    -----------
    x0 : 1D numpy array
        initial estimate of solution vector
    a_m : function
        function that accepts 1D numpy array of the approx. solution vector
        and returns the Jacobian matrix of the funcitons (a 2D numpy array 
        of the partial derivatives evaluated at the approx. solution vector)
    b_v : function
        function that accepts 1D numpy array of the approx. solution vector
        and returns a 2D numpy array of the functions evaluated at the approx.
        solution vector)
    num_eqs : integer
        number of equations in the system
    n : integer
        number of iterations to use the same Jacobian w/o update (a good starting
        point is the number of functions in the system)
    tol : float, optional
        convergence tolerance. The default is 0.0001
    max_iter : integer, optional
        maximum number of iterations to perform if the solution is not converging.
        The default is 20.
    verbose : bool, optional
        outputs a summary of the solution progression at each solution step.
        The default is False.
        
    Returns:
    --------
    x0 : 1D numpy array
        approx. solution to the system within the specified tolerance
    dx : 1D numpy array
        solution error w.r.t. the previous approx. solution
    num_iter : integer
        number of iterations required to converge.
        
    Notes:
    ------
    Example a_m function:
    
    def a_m(z):
        x = z[0]
        y = z[1]

        dxf1 = lambda x,y: 2.0*x + 1.0
        dyf1 = lambda x,y: -2.0*y
        dxf2 = lambda x,y: -2.0*x*math.cos(math.pow(x,2))
        dyf2 = lambda x,y: 1.0

        return np.array([[dxf1(x,y),dyf1(x,y)],[dxf2(x,y),dyf2(x,y)]])
    
    Example b_m function:
    
    def b_v(z):
        x = z[0]
        y = z[1]

        f1 = lambda x,y: math.pow(x,2) + x - math.pow(y,2) - 1
        f2 = lambda x,y: y - math.sin(math.pow(x,2))
        return np.array([f1(x,y),f2(x,y)])   
    """
    
    tol = np.full(num_eqs, 0.0001)
    dx = tol*10
    num_iter = 0
    max_iter = 20

    while np.all(abs(dx)>tol) and num_iter < max_iter:
        a = a_m(x0)
        i = 0
        
        while np.all(abs(dx)>tol) and i < n and num_iter < max_iter:
            b = -1*b_v(x0)

            lu, ov = gedo(a)
            dx = dosv(lu, ov, b)[ov]

            x0 = x0 + dx
            
            i += 1
            num_iter += 1

            if verbose:
                print(f'a: {a}')
                print(f'b: {b}')
                print(f'lu,ov: {lu},{ov}')
                print(f'dx: {dx}')
                print(f'x0: {x0}')
                print(f'num_iter: {num_iter}')
                print('\n')

    return x0, dx, num_iter
