import numpy as np

def lu_decomp(A, b):
n,m = A.shape
ov = np.arange(n) #order_vector

if n != m:
    raise ValueError("Matrix is not sqaure.")

#row reduction
for j in xrange(n):
    pvt = abs(A[ov[j],j]) #gets current pvt on diagonal
    org_pvt_row = j #keeps track of row location of pvt

    #cycle thru entries in first column
    #find largest value
    for i in xrange(j+1,n):
        if abs(A[i,j]) > pvt:
            pvt = abs(A[i,j])
            new_pvt_row = i

    #switch largest value to be pivot
    #this reduces rounding error
    if org_pvt_row != new_pvt_row:
        ov[org_pvt_row] = org_pvt_row
        ov[new_pvt_row] = new_pvt_row

    #calculates multipliers for row reduction
    for i in xrange(j+1,n):
        A[ov[i],j] = A[ov[i],j]/A[ov[j],j]

    #creates zeros below the main diagonal
    for i in xrange(j+1,n): #steps across columns
        for k in xrange(j+1,n): #steps down rows in each column
            A[ov[i],k] = A[ov[i],k] - A[ov[i],j]*A[ov[j],k]
        b[ov[i]] = b[ov[i]] - a[ov[i],j]*b[ov[j]] #performs rhs calc

    return A, b, ov #where A is now L in the lower and U in the upper

def back_sub(A, b):
#back substitution
x[ov[n]] = b[ov[n]]/A[ov[n],n] # last row solution
for j in xrange(n-1,1,-1):
    x[ov[j]] = b[ov[j]] #initialize solution value
    for k in xrange(n,j+1,-1):
        #group known terms in numerator
        x[ov[j]] = x[ov[j]] - x[ov[k]]*A[ov[j],k] 
    x[ov[j]] = x[ov[j]]/A[ov[j],j] #solving equation by division

    return x

def multiple_rhs():
    #function to handle multiplication of the LU matricies, order vector and
    #back substitution after the LU decomposition has been done the first time.

    #one may want to incorporate this into the typical flow of this program
    #i.e. have one fuction to calc the LU decomposition, one function to apply
    #the LU decomposition to the rhs, and one function to complete the back
    #substituation to solve the equations; not sure if this will be slow than
    #having a seperate function for additional rhs as currently shown
    pass
