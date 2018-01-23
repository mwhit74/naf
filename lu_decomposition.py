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

def back_sub():
#back substitution
x[ov[n]] = b[ov[n]]/a[ov[n],n] # last row solution
for j in xrange(n-1,1,-1):
    x[ov[j]] = b[ov[j]]
    for k in xrange(n,j+1,-1):
        x[ov[j]] = x[ov[j]] - x[ov[k]]*a[ov[j],k]
    x[ov[j]] = x[ov[j]]/a[ov[j],j]

