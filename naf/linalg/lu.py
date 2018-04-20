import numpy as np
import pdb


def lu_decomp(A):
    """Finds the LU decomposition of the matrix A.

    This function factors the A matrix into two matricies L and U. L is a lower
    triangular matrix of the multiplication coefficients with 1's on the
    diagonal. U is an upper triangular matrix of the remaining coefficients of
    the original A matrix after pivoting and row reduction.

    This function does the decomposition in place. That is to say it stores the
    resulting values in the A matrix. 

    This function also keeps track of any row exchanges due to partial pivoting
    of the rows. 

    Args:
        A (two dimensional numpy array): matrix to be factored into LU

    Retuns:
        A (two dimensional numpy array): combined L and U matricies; L is the
            lower triangular with 1's on the diagonal; U is the upper triangular
            including the values on the diagonal
        ov (one dimensional numpy array): the permutated order of the rows after
            partial pivoting and factorization

    """
    n,m = A.shape
    #n minus one, shape returns the number of elements; not good for
    #indexing
    nmo = n - 1
    ov = np.arange(n) #order_vector
    
    if n != m:
        raise ValueError("Matrix is not sqaure.")
    
    #row reduction
    for j in xrange(nmo):
        pvt = abs(A[ov[j],j]) #gets current pvt on diagonal
        new_pvt_row = None
        org_pvt_row = j #keeps track of row location of pvt
    
        #cycle thru entries in first column
        #find largest value
        #aka finding the max pivot
        for i in xrange(j+1,n):
            if abs(A[ov[i],j]) > pvt:
                pvt = abs(A[ov[i],j])
                new_pvt_row = i
    
        #switch largest value to be pivot
        #this reduces rounding error
        if (new_pvt_row != None and org_pvt_row != new_pvt_row):
            ov[org_pvt_row] = new_pvt_row
            ov[new_pvt_row] = org_pvt_row
    
        #calculates multipliers for row reduction
        for i in xrange(j+1,n):
            A[ov[i],j] = A[ov[i],j]/A[ov[j],j]
    
        #creates zeros below the main diagonal
        for i in xrange(j+1,n): #row number
            for k in xrange(j+1,n): #column number
                A[ov[i],k] = A[ov[i],k] - A[ov[i],j]*A[ov[j],k]
    
    return A, ov #where A is now L in the lower and U in the upper


def for_elim(LU, ov, b):
    """Forward elmination applied to the b vector.

    Applies the operations recorded in the decomposition, the L lower triangular
    matrix and the row exchanges, to the b vector. 

    This step can be taken in the LU decomposition function but there are
    significant efficienies to be gained by having a separate function if there
    are multiple b vectors to be considered which is very often the case. 

    Args:
        LU (two dimensional numpy array): factored A matrix; L is the lower
            triangular with 1's on the diagonal; U is the upper triangular including
            the values on the diagonal
        b (one dimensional numpy array): the right hand side; the constant
            values in the equations
        ov (one dimensional numpy array): the permutated order of the rows after
            partial pivoting and factorization

    Returns:
        b (one dimensional numpy array): often referred to 'c' in literature on
            the subject; the b vector with the factored operations from the
            decomposition applied

    """
    n,m = LU.shape
    nmo = n - 1
    for j in xrange(nmo):
        for i in xrange(j+1,n):
            b[ov[i]] = b[ov[i]] - LU[ov[i],j]*b[ov[j]]

    return b


def back_sub(LU, ov, b):
    """
    Back substituation applied to solve for the x vector.

    Solves the upper triangular matrix U with the factored b vector for the x
    vector. 

    Args:
        LU (two dimensional numpy array): factored A matrix; L is the lower
            triangular with 1's on the diagonal; U is the upper triangular including
            the values on the diagonal
        b (one dimensional numpy array): often referred to 'c' in literature on
            the subject; the b vector with the factored operations from the
            decomposition applied
        ov (one dimensional numpy array): the permutated order of the rows after
            partial pivoting and factorization

    Retuns:
        x (one dimensional numpy array): the solution vector
    
    """
    n,m = LU.shape
    nmo = n - 1
    x = np.empty((n,1),dtype='float')   
    #back substitution
    x[ov[nmo]] = b[ov[nmo]]/LU[ov[nmo],nmo] # last row solution
    for j in xrange(nmo-1,-1,-1):
        x[ov[j]] = b[ov[j]] #initialize solution value
        for k in xrange(nmo,j,-1):
            #group known terms in numerator
            x[ov[j]] = x[ov[j]] - x[ov[k]]*LU[ov[j],k] 
        x[ov[j]] = x[ov[j]]/LU[ov[j],j] #solving equation by division
    
    return x


def lu_solve(LU, ov, b):
    """Applies the forward elimination and back substituation steps."""
    b = for_elim(LU, ov, b)
    x = back_sub(LU, ov, b)
    return x


if __name__ == "__main__":
    #A1 = np.array([[0.0,2.0,0.0,1.0],
    #              [2.0,2.0,3.0,2.0],
    #              [4.0,-3.0,0.0,1.0],
    #              [6.0,1.0,-6.0,-5.0]])
    A1 = np.array([[696.0, 0.0],
                  [0.0, 2142.25]])
    print A1
    #b1 = np.array([[0.0],
    #              [-2.0],
    #              [-7.0],
    #              [6.0]])
    b1 = np.array([[150.0],
                  [-300.0]])
    print b1
    pdb.set_trace()
    LU, ov = lu_decomp(A1)
    print LU
    print ov
    x = lu_solve(LU, ov, b1)
    print x
