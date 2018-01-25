import numpy as np
import pdb

def lu_decomp(A, b):
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
        if org_pvt_row != new_pvt_row:
            ov[org_pvt_row] = new_pvt_row
            ov[new_pvt_row] = org_pvt_row
    
        #calculates multipliers for row reduction
        for i in xrange(j+1,n):
            A[ov[i],j] = A[ov[i],j]/A[ov[j],j]
    
        #creates zeros below the main diagonal
        for i in xrange(j+1,n): #row number
            for k in xrange(j+1,n): #column number
                A[ov[i],k] = A[ov[i],k] - A[ov[i],j]*A[ov[j],k]
            b[ov[i]] = b[ov[i]] - A[ov[i],j]*b[ov[j]] #performs rhs calc
    
    return A, b, ov #where A is now L in the lower and U in the upper

def back_sub(A, b, ov):
    n,m = A.shape
    nmo = n - 1
    x = np.empty((n,1),dtype='float')   
    #back substitution
    x[ov[nmo]] = b[ov[nmo]]/A[ov[nmo],nmo] # last row solution
    for j in xrange(nmo-1,-1,-1):
        x[ov[j]] = b[ov[j]] #initialize solution value
        for k in xrange(nmo,j,-1):
            #group known terms in numerator
            x[ov[j]] = x[ov[j]] - x[ov[k]]*A[ov[j],k] 
        x[ov[j]] = x[ov[j]]/A[ov[j],j] #solving equation by division
    
    return x

#def rhs(A, b, ov):
#    #one may want to incorporate this into the typical flow of this program
#    #i.e. have one fuction to calc the LU decomposition, one function to apply
#    #the LU decomposition to the rhs, and one function to complete the back
#    #substituation to solve the equations; not sure if this will be slow than
#    #having a seperate function for additional rhs as currently shown
#    n,m = A.shape
#    nmo = n - 1
#    for j in xrange(nmo):
#        for i in xrange(j+1,n):
#            b[ov[i]] = b[ov[i]] - A[ov[i],j]*b[ov[j]]
#
#    return b
#def forward_sub():
#
#def lu_solve(A, b):
#    A, ov = lu_decomp(A)
#    b = rhs(A, b, ov)
#    x = back_sub(A, b, ov)

if __name__ == "__main__":
    A = np.array([[0.0,2.0,0.0,1.0],
                  [2.0,2.0,3.0,2.0],
                  [4.0,-3.0,0.0,1.0],
                  [6.0,1.0,-6.0,-5.0]])
    print A
    b = np.array([[0.0],
                  [-2.0],
                  [-7.0],
                  [6.0]])
    print b
    A,b,ov = lu_decomp(A, b)
    x = back_sub(A, b, ov)
    print A
    print ov
    print b
    print x
